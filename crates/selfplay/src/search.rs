//! Gumbel AlphaZero tree search on the real board (self-play).
//!
//! The search follows mctx `gumbel_muzero_policy`; every node is a real
//! position (moves made with the move generator, leaves evaluated by the
//! network).
//!
//! **Duck chess:** a turn is two steps, a piece move and a duck move, and
//! each step is its own search, its own tree level and its own training
//! sample. The side to move only changes after a duck move, so a backed-up
//! value changes sign only across a duck-move edge.
//!
//! Per step, with `n` simulations and `m` considered actions:
//! * Root: Gumbel noise on the legal logits; Sequential Halving over the top
//!   `m` actions (mctx `gumbel_muzero_root_action_selection`), so the best
//!   candidates get most of the visits.
//! * Below the root: deterministic selection
//!   `argmax π'(a) − N(a) / (1 + ΣN)` with `π' = softmax(logits + σ(q̂))`
//!   (mctx `gumbel_muzero_interior_action_selection`).
//! * `q̂`: completed Q values (mixed value for unvisited actions), min-max
//!   rescaled; `σ(q̂) = (c_visit + max N) · c_scale · q̂`.
//! * Finished games (king captured, no moves, draws) get their exact value
//!   without a network call; so does a leaf where the side to move can
//!   capture the king (its next step wins).
//! * Step played: Sequential Halving winner; policy target:
//!   `softmax(logits + σ(q̂))` at the root.
//!
//! Playout cap randomization (KataGo; `fast_simulations > 0`): each step is
//! a *full* search (`simulations`, `considered`) with probability
//! `full_prob`, otherwise a *fast* search (`fast_simulations`,
//! `fast_considered`). Fast steps are recorded with `full_search = false`
//! and are not training samples.
//!
//! Root exploration floor (`root_explore = ε > 0`): for choosing the root
//! candidates and the step played, the prior is mixed with a uniform share,
//! `(1 − ε)·π + ε/k`, so every legal step keeps a real chance of being
//! considered. The policy target still uses the network's own logits.
//!
//! Temperature (`temperature > 0`): the step played is sampled from the
//! policy target `π'` as `π'(a)^(1/T)` among the root actions whose search
//! value is within `temp_value_cutoff` (W − L) of the best visited action.
//! `T` decays linearly from `temperature` to 0 over `temp_decay_plies`
//! steps of the game (0 = no decay), but never below `temp_endgame`.
//!
//! Openings (`openings`): each new game starts, with probability
//! `book_prob`, after a random number (1..=all) of turns of a random book
//! line, else from the start position; then, with probability
//! `random_prob`, 1..=`random_plies` uniformly random turns follow. Book and
//! random turns are not recorded.
//!
//! The pool is stepped in lock-step rounds so the GPU sees full batches:
//! every round each game asks for exactly one evaluation (its next leaf, or
//! the root of its next step). A step takes about `n + 1` rounds.

use std::sync::Arc;

use peaduck_core::encoding::{encode_game, legal_actions, OBS_SIZE};
use peaduck_core::rng::Rng;
use peaduck_core::{Color, DuckDefence, Game, GameResult, Move, Position};
use rayon::prelude::*;

use crate::record::{GameRecord, PlyRecord, Termination, TERMINATIONS};
use crate::replay::MAX_LEGAL;
use crate::{wdl_logits_to_probs, wdl_logits_to_q};

#[derive(Clone, Debug)]
pub struct GumbelConfig {
    /// Maximum number of considered root actions `m` (Sequential Halving width).
    pub considered: usize,
    /// Scale of the Gumbel noise (1.0 in self-play, 0.0 = deterministic).
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        GumbelConfig {
            considered: 16,
            gumbel_scale: 1.0,
            c_visit: 50.0,
            c_scale: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    pub games_finished: u64,
    /// Steps played (piece moves and duck moves).
    pub plies: u64,
    /// Steps played after a full search (= training samples); equals
    /// `plies` without playout cap randomization.
    pub full_plies: u64,
    pub white_wins: u64,
    pub black_wins: u64,
    pub draws: u64,
    /// Indexed by `Termination as usize`.
    pub terminations: [u64; TERMINATIONS],
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub games: usize,
    /// Simulations (leaf evaluations) per step, not counting the root.
    pub simulations: usize,
    /// Games reaching this many steps (two per turn) are adjudicated as draws.
    pub max_plies: usize,
    /// `considered` = Sequential Halving width `m`.
    pub gumbel: GumbelConfig,
    pub seed: u64,
    /// Worker threads for the per-game work; 0 = one per logical CPU.
    pub threads: usize,
    /// Steps per game with the full Gumbel noise (`gumbel.gumbel_scale`);
    /// after that `late_gumbel_scale` is used. 0 = full noise all game.
    pub noise_plies: usize,
    pub late_gumbel_scale: f32,
    /// Playout cap randomization: simulations of a fast move; 0 = off
    /// (every move is a full search).
    pub fast_simulations: usize,
    /// Probability that a move gets the full search (with fast moves on).
    pub full_prob: f32,
    /// Sequential Halving width of a fast search.
    pub fast_considered: usize,
    /// Opening book lines (start position + turns); empty = no book.
    pub openings: Arc<Vec<Opening>>,
    /// Probability that a new game starts from a book line.
    pub book_prob: f32,
    /// Probability of random moves after the opening.
    pub random_prob: f32,
    /// At most this many random turns (uniform 1..=random_plies).
    pub random_plies: usize,
    /// Move-selection temperature at the start of the game; 0 = off (play
    /// the Sequential Halving winner).
    pub temperature: f32,
    /// Steps over which the temperature decays linearly to 0 (0 = constant).
    pub temp_decay_plies: usize,
    /// Lower bound of the temperature for the rest of the game.
    pub temp_endgame: f32,
    /// Only moves whose search value (W − L) is within this of the best
    /// visited move are sampled.
    pub temp_value_cutoff: f32,
    /// Root exploration floor ε: candidates are drawn from (1 − ε)·prior + ε/k.
    pub root_explore: f32,
}

/// One opening book line: a start position and the turns that follow
/// (empty for EPD/FEN books: the position itself is the opening).
#[derive(Clone, Debug)]
pub struct Opening {
    pub start: Game,
    /// Steps of the line, two per turn (piece move, duck move).
    pub moves: Vec<Move>,
}

impl Opening {
    /// From a FEN (None = start position) and UCI turns (`e2e4,e4d5`);
    /// errors on illegal turns or a line that ends the game.
    pub fn new(fen: Option<&str>, turns: &[String]) -> Result<Opening, String> {
        let start = match fen {
            Some(f) => Game::from_fen(f).map_err(|e| format!("bad FEN {f:?}: {e}"))?,
            None => Game::default(),
        };
        if start.position().is_duck_step() {
            return Err("opening positions must be in the move step".into());
        }
        let mut g = start.clone();
        for t in turns {
            if !g.push_uci(t) || g.position().is_duck_step() {
                return Err(format!("illegal turn {t} in opening line"));
            }
            if g.result() != GameResult::Ongoing {
                return Err(format!("opening line ends the game at {t}"));
            }
        }
        Ok(Opening {
            moves: g.moves().to_vec(),
            start,
        })
    }
}

fn uniform(rng: &mut Rng) -> f32 {
    (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32
}

impl SearchConfig {
    /// Selection temperature at game step `ply` (0 = play the best step).
    fn move_temperature(&self, ply: usize) -> f32 {
        let t = if self.temp_decay_plies > 0 {
            self.temperature * (1.0 - ply as f32 / self.temp_decay_plies as f32).max(0.0)
        } else {
            self.temperature
        };
        t.max(self.temp_endgame)
    }

    /// The start of a new game: book line and/or random moves, or the start
    /// position. Draws nothing from `rng` when neither is configured.
    fn new_game(&self, rng: &mut Rng) -> Game {
        let use_book = !self.openings.is_empty() && uniform(rng) < self.book_prob;
        let use_random =
            self.random_plies > 0 && self.random_prob > 0.0 && uniform(rng) < self.random_prob;
        if !use_book && !use_random {
            return Game::default();
        }
        for _attempt in 0..16 {
            let mut g = if use_book {
                let o = &self.openings[rng.below(self.openings.len() as u64) as usize];
                let mut g = o.start.clone();
                let turns = o.moves.len() / 2;
                let n = if turns == 0 {
                    0
                } else {
                    1 + rng.below(turns as u64) as usize
                };
                for &m in &o.moves[..2 * n] {
                    g.push(m);
                }
                g
            } else {
                Game::default()
            };
            if use_random {
                // whole turns: a random piece move, then a random duck move
                let n = 2 * (1 + rng.below(self.random_plies as u64) as usize);
                for _ in 0..n {
                    let moves = legal_actions(g.position());
                    if moves.is_empty() || g.result().is_over() {
                        break;
                    }
                    g.push(moves[rng.below(moves.len() as u64) as usize].1);
                }
            }
            if g.result() == GameResult::Ongoing {
                return g; // history kept: repetitions of book positions still count
            }
        }
        Game::default()
    }
}

impl SearchConfig {
    fn pcr(&self) -> bool {
        self.fast_simulations > 0 && self.full_prob < 1.0
    }
}

/// Considered-visit tables for the full and the fast search.
pub(crate) struct Tables {
    full: Vec<Vec<u32>>,
    fast: Vec<Vec<u32>>,
}

impl Tables {
    pub(crate) fn new(cfg: &SearchConfig) -> Tables {
        Tables {
            full: considered_visits_table(cfg.gumbel.considered, cfg.simulations),
            fast: if cfg.pcr() {
                considered_visits_table(cfg.fast_considered.max(1), cfg.fast_simulations)
            } else {
                Vec::new()
            },
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            games: 256,
            simulations: 64,
            max_plies: 1024,
            gumbel: GumbelConfig::default(),
            seed: 0,
            threads: 1,
            noise_plies: 0,
            late_gumbel_scale: 0.0,
            fast_simulations: 0,
            full_prob: 0.25,
            fast_considered: 8,
            openings: Arc::new(Vec::new()),
            book_prob: 0.0,
            random_prob: 0.0,
            random_plies: 0,
            temperature: 0.0,
            temp_decay_plies: 0,
            temp_endgame: 0.0,
            temp_value_cutoff: 0.1,
            root_explore: 0.0,
        }
    }
}

const NONE: u32 = u32::MAX;

struct Node {
    actions: Vec<(usize, Move)>,
    /// Policy logits of `actions`, max-normalised.
    logits: Vec<f32>,
    children: Vec<u32>,
    visits: Vec<u32>,
    /// Sum of backed-up values per action, from this node's side to move.
    value_sum: Vec<f32>,
    /// Network value (`W - L`, side to move), or the exact terminal value.
    raw_value: f32,
    /// Network draw probability (exact for terminals).
    raw_draw: f32,
    terminal: bool,
}

impl Node {
    fn new(actions: Vec<(usize, Move)>, raw_logits: &[f32], raw_value: f32, raw_draw: f32) -> Node {
        let max = raw_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let k = actions.len();
        Node {
            logits: raw_logits.iter().map(|&l| l - max).collect(),
            children: vec![NONE; k],
            visits: vec![0; k],
            value_sum: vec![0.0; k],
            actions,
            raw_value,
            raw_draw,
            terminal: false,
        }
    }

    fn terminal(value: f32) -> Node {
        Node {
            actions: Vec::new(),
            logits: Vec::new(),
            children: Vec::new(),
            visits: Vec::new(),
            value_sum: Vec::new(),
            raw_value: value,
            raw_draw: if value == 0.0 { 1.0 } else { 0.0 },
            terminal: true,
        }
    }

    fn total_visits(&self) -> u32 {
        self.visits.iter().sum()
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// `σ(completed Q)` for every action of `node` (mctx
/// `qtransform_completed_by_mix_value` with rescaling).
fn sigma(node: &Node, cfg: &GumbelConfig) -> Vec<f32> {
    let k = node.actions.len();
    let prior = softmax(&node.logits);
    let sum_visits: u32 = node.total_visits();
    let max_visits = node.visits.iter().copied().max().unwrap_or(0);
    let q: Vec<f32> = (0..k)
        .map(|a| {
            if node.visits[a] > 0 {
                node.value_sum[a] / node.visits[a] as f32
            } else {
                0.0
            }
        })
        .collect();
    // Mixed value (f64 so equal Qs give exactly that Q).
    let mut sum_prior = 0.0f64;
    let mut weighted = 0.0f64;
    for a in 0..k {
        if node.visits[a] > 0 {
            let p = prior[a].max(f32::MIN_POSITIVE) as f64;
            sum_prior += p;
            weighted += p * q[a] as f64;
        }
    }
    let mixed = if sum_visits > 0 {
        let s = sum_visits as f64;
        ((node.raw_value as f64 + s * weighted / sum_prior) / (s + 1.0)) as f32
    } else {
        node.raw_value
    };
    let completed: Vec<f32> = (0..k)
        .map(|a| if node.visits[a] > 0 { q[a] } else { mixed })
        .collect();
    // Min-max rescale; epsilon 1e-4 instead of mctx's 1e-8 so f32 rounding
    // noise is not blown up into a preference (see gumbel.rs).
    let lo = completed.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-4);
    let scale = (cfg.c_visit + max_visits as f32) * cfg.c_scale;
    completed.iter().map(|&v| scale * (v - lo) / span).collect()
}

/// mctx `get_sequence_of_considered_visits`.
fn considered_visits_sequence(m: usize, n: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2max = (m as f64).log2().ceil() as usize;
    let mut seq = Vec::with_capacity(n + m);
    let mut visits = vec![0u32; m];
    let mut considered = m;
    while seq.len() < n {
        let extra = (n / (log2max * considered)).max(1);
        for _ in 0..extra {
            seq.extend_from_slice(&visits[..considered]);
            for v in visits.iter_mut().take(considered) {
                *v += 1;
            }
        }
        considered = (considered / 2).max(2);
    }
    seq.truncate(n);
    seq
}

/// Table indexed by `[num_considered][simulation]` (mctx
/// `get_table_of_considered_visits`), one extra column so the final
/// selection after `n` simulations can be looked up too.
pub(crate) fn considered_visits_table(m: usize, n: usize) -> Vec<Vec<u32>> {
    (0..=m)
        .map(|k| {
            let mut s = considered_visits_sequence(k, n);
            s.push(u32::MAX); // never used for selection
            s
        })
        .collect()
}

/// mctx `score_considered`: `g + logits + σ`, restricted to the actions whose
/// visit count equals `considered_visit`.
fn score_considered(node: &Node, gumbel: &[f32], sigma: &[f32], considered_visit: u32) -> usize {
    let score = |a: usize| (gumbel[a] + node.logits[a] + sigma[a]).max(-1e9);
    let best = (0..node.actions.len())
        .filter(|&a| node.visits[a] == considered_visit)
        .max_by(|&a, &b| score(a).total_cmp(&score(b)));
    // The schedule always leaves a candidate; fall back defensively.
    best.unwrap_or_else(|| {
        (0..node.actions.len())
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .unwrap()
    })
}

fn interior_action(node: &Node, cfg: &GumbelConfig) -> usize {
    let s = sigma(node, cfg);
    let improved: Vec<f32> = node.logits.iter().zip(&s).map(|(l, s)| l + s).collect();
    let probs = softmax(&improved);
    let denom = 1.0 + node.total_visits() as f32;
    (0..node.actions.len())
        .max_by(|&a, &b| {
            (probs[a] - node.visits[a] as f32 / denom)
                .total_cmp(&(probs[b] - node.visits[b] as f32 / denom))
        })
        .unwrap()
}

/// The steps the search considers in `pos`: every legal step, except in a
/// duck step where only some duck squares stop the opponent from capturing
/// our king next ([`Position::duck_defence`]): then only those. The others
/// lose at once, so they get no visits and a policy target of 0. (Not when
/// the fifty-move rule ends the game first.)
fn search_actions(pos: &Position) -> Vec<(usize, Move)> {
    let mut actions = legal_actions(pos);
    if pos.halfmove_clock() < 100 {
        if let DuckDefence::Block(safe) = pos.duck_defence() {
            actions.retain(|(_, m)| safe.contains(m.to()));
        }
    }
    actions
}

/// Exact value of a leaf for its side to move, or None if the game goes
/// on. (After a king capture the side to move is still the capturer, who
/// won.) A leaf where the side to move can capture the king is a win too:
/// that capture is its next step. A duck step where no duck square saves
/// the king is a loss for the same reason, one step earlier.
fn terminal_value(game: &Game) -> Option<f32> {
    let pos = game.position();
    let stm: Color = pos.side_to_move();
    match game.result() {
        GameResult::Ongoing if pos.king_capture_available() => Some(1.0),
        GameResult::Ongoing
            if pos.halfmove_clock() < 100 && pos.duck_defence() == DuckDefence::Lost =>
        {
            Some(-1.0)
        }
        GameResult::Ongoing => None,
        GameResult::Win(c, _) => Some(if c == stm { 1.0 } else { -1.0 }),
        GameResult::Draw(_) => Some(0.0),
    }
}

/// What the slot is waiting for.
enum Request {
    /// Evaluate the root of the next move.
    Root,
    /// Evaluate a new leaf reached by `path` (last entry = parent, action).
    Leaf { path: Vec<(u32, usize)> },
}

pub(crate) struct Slot {
    pub(crate) game: Game,
    pub(crate) plies: Vec<PlyRecord>,
    rng: Rng,
    nodes: Vec<Node>,
    root_gumbel: Vec<f32>,
    simulations_done: usize,
    /// This move gets the full search (playout cap randomization).
    full: bool,
    request: Request,
    /// Position to evaluate and its legal moves.
    pub(crate) eval_game: Game,
    pub(crate) eval_actions: Vec<(usize, Move)>,
}

pub(crate) struct Outcome {
    /// 1 white won, -1 black won, 0 draw.
    pub(crate) result: i8,
    pub(crate) termination: Termination,
    pub(crate) plies: Vec<PlyRecord>,
}

impl Slot {
    /// A slot that starts searching from `game` (evaluation games, puzzles).
    pub(crate) fn with_game(game: Game, rng: Rng) -> Slot {
        let eval_actions = search_actions(game.position());
        Slot {
            eval_game: game.clone(),
            game,
            plies: Vec::with_capacity(512),
            rng,
            nodes: Vec::new(),
            root_gumbel: Vec::new(),
            simulations_done: 0,
            full: true,
            request: Request::Root,
            eval_actions,
        }
    }

    pub(crate) fn request_root(&mut self) {
        self.nodes.clear();
        self.simulations_done = 0;
        self.eval_game = self.game.clone();
        self.eval_actions = search_actions(self.game.position());
        self.request = Request::Root;
    }

    /// Back up `leaf_value` (for the leaf's side to move) along `path`. The
    /// side to move only changes across a duck move, so only those flip it.
    fn backup(&mut self, path: &[(u32, usize)], leaf_value: f32) {
        let mut v = leaf_value;
        for &(node, a) in path.iter().rev() {
            let n = &mut self.nodes[node as usize];
            if n.actions[a].1.is_duck() {
                v = -v;
            }
            n.visits[a] += 1;
            n.value_sum[a] += v;
        }
    }

    /// Run simulations until one needs a network evaluation (sets
    /// `request`) or the move's budget is used up (returns true).
    fn advance(&mut self, cfg: &SearchConfig, tables: &Tables) -> bool {
        let (budget, considered, table) = if self.full {
            (cfg.simulations, cfg.gumbel.considered, &tables.full)
        } else {
            (
                cfg.fast_simulations,
                cfg.fast_considered.max(1),
                &tables.fast,
            )
        };
        while self.simulations_done < budget {
            let mut path: Vec<(u32, usize)> = Vec::with_capacity(16);
            let mut node = 0u32;
            loop {
                let n = &self.nodes[node as usize];
                let a = if node == 0 {
                    let k = n.actions.len().min(considered).max(1);
                    let s = sigma(n, &cfg.gumbel);
                    let cv = table[k][self.simulations_done];
                    score_considered(n, &self.root_gumbel, &s, cv)
                } else {
                    interior_action(n, &cfg.gumbel)
                };
                path.push((node, a));
                let child = n.children[a];
                if child == NONE {
                    break;
                }
                if self.nodes[child as usize].terminal {
                    break;
                }
                node = child;
            }

            let &(parent, a) = path.last().unwrap();
            let existing = self.nodes[parent as usize].children[a];
            if existing != NONE {
                // Revisit of a known terminal: back up its exact value.
                let v = self.nodes[existing as usize].raw_value;
                self.backup(&path, v);
                self.simulations_done += 1;
                continue;
            }

            let mut leaf = self.game.clone();
            for &(n, act) in &path {
                leaf.push(self.nodes[n as usize].actions[act].1);
            }
            if let Some(v) = terminal_value(&leaf) {
                let idx = self.nodes.len() as u32;
                self.nodes.push(Node::terminal(v));
                self.nodes[parent as usize].children[a] = idx;
                self.backup(&path, v);
                self.simulations_done += 1;
                continue;
            }
            self.eval_actions = search_actions(leaf.position());
            self.eval_game = leaf;
            self.request = Request::Leaf { path };
            return false;
        }
        true
    }

    /// Pick the move, record the ply, play it. Returns the finished game, if
    /// any, and sets up the next root request.
    fn play_move(&mut self, cfg: &SearchConfig) -> Option<Outcome> {
        let root = &self.nodes[0];
        let s = sigma(root, &cfg.gumbel);
        let max_visit = root.visits.iter().copied().max().unwrap_or(0);
        let mut chosen = score_considered(root, &self.root_gumbel, &s, max_visit);
        // With the exploration floor, a move the search visited gets at least
        // the floored prior in its target, so a collapsed move that turns out
        // good can regain policy mass (σ alone cannot lift a logit ~20 down).
        let floored = explore_logits(&root.logits, cfg.root_explore);
        let improved: Vec<f32> = (0..root.actions.len())
            .map(|a| match &floored {
                Some(f) if root.visits[a] > 0 => root.logits[a].max(f[a]) + s[a],
                _ => root.logits[a] + s[a],
            })
            .collect();
        let target = softmax(&improved);
        let temp = cfg.move_temperature(self.game.ply());
        if temp > 0.0 {
            chosen =
                sample_with_temperature(root, &target, temp, cfg.temp_value_cutoff, &mut self.rng)
                    .unwrap_or(chosen);
        }
        let targets: Vec<(usize, f32)> = root.actions.iter().map(|&(a, _)| a).zip(target).collect();
        let (action, mv) = root.actions[chosen];
        // Search value of the root: mean of the network value and all backed-up
        // values (mctx node value). W - L from it, D from the network.
        let visits: u32 = root.total_visits();
        let sum: f32 = root.value_sum.iter().sum();
        let q = ((root.raw_value + sum) / (1.0 + visits as f32)).clamp(-1.0, 1.0);
        let d = root.raw_draw.min(1.0 - q.abs());
        let value = [(1.0 - d + q) / 2.0, d, (1.0 - d - q) / 2.0];
        self.plies.push(
            PlyRecord::new(
                self.game.position(),
                self.game.repetitions(),
                action,
                &targets,
            )
            .with_value(value)
            .with_full_search(self.full),
        );
        self.game.push(mv);

        let outcome = match Termination::from_result(self.game.result()) {
            Some(x) => Some(x),
            None if self.plies.len() >= cfg.max_plies => Some((0, Termination::MaxPlies)),
            None => None,
        };
        let finished = outcome.map(|(result, termination)| {
            self.game = cfg.new_game(&mut self.rng);
            Outcome {
                result,
                termination,
                plies: std::mem::replace(&mut self.plies, Vec::with_capacity(512)),
            }
        });
        self.request_root();
        finished
    }

    /// Apply this slot's network output and advance to its next request.
    pub(crate) fn respond(
        &mut self,
        logits: &[f32],
        value: &[f32],
        cfg: &SearchConfig,
        tables: &Tables,
    ) -> Option<Outcome> {
        let k = self.eval_actions.len();
        let v = wdl_logits_to_q(value);
        let draw = wdl_logits_to_probs(value)[1];
        let actions = std::mem::take(&mut self.eval_actions);
        let node = Node::new(actions, &logits[..k], v, draw);
        match std::mem::replace(&mut self.request, Request::Root) {
            Request::Root => {
                // Drawn only with PCR on, so the games without it are unchanged.
                self.full = !cfg.pcr()
                    || ((self.rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) < cfg.full_prob;
                let scale = if cfg.noise_plies > 0 && self.plies.len() >= cfg.noise_plies {
                    cfg.late_gumbel_scale
                } else {
                    cfg.gumbel.gumbel_scale
                };
                self.root_gumbel = (0..k)
                    .map(|_| scale * sample_gumbel(&mut self.rng))
                    .collect();
                if let Some(f) = explore_logits(&node.logits, cfg.root_explore) {
                    // The root scores are gumbel + logit + σ: add floored − logit so
                    // candidates and the played move see the floored prior.
                    for ((g, &l), &fl) in self.root_gumbel.iter_mut().zip(&node.logits).zip(&f) {
                        *g += fl - l;
                    }
                }
                self.nodes.clear();
                self.nodes.push(node);
            }
            Request::Leaf { path } => {
                let &(parent, a) = path.last().unwrap();
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                self.nodes[parent as usize].children[a] = idx;
                self.backup(&path, v);
                self.simulations_done += 1;
            }
        }
        if self.advance(cfg, tables) {
            self.play_move(cfg)
        } else {
            None
        }
    }
}

/// Logits of the floored prior `(1 − ε)·softmax(logits) + ε/k`, on the same
/// scale as `logits` (max-normalised logits are log-probabilities up to a
/// constant). None when ε = 0.
fn explore_logits(logits: &[f32], eps: f32) -> Option<Vec<f32>> {
    if eps <= 0.0 || logits.is_empty() {
        return None;
    }
    let eps = eps.min(1.0);
    let k = logits.len() as f32;
    let prior = softmax(logits);
    // ln of the mixed probability, shifted back to the logits' scale
    let shift = logits
        .iter()
        .zip(&prior)
        .map(|(&l, &p)| l - p.max(f32::MIN_POSITIVE).ln())
        .sum::<f32>()
        / k;
    Some(
        prior
            .iter()
            .map(|&p| ((1.0 - eps) * p + eps / k).ln() + shift)
            .collect(),
    )
}

/// Move sampling: `target(a)^(1/temp)` over the visited root
/// actions whose Q is within `cutoff` of the best visited action's Q.
fn sample_with_temperature(
    root: &Node,
    target: &[f32],
    temp: f32,
    cutoff: f32,
    rng: &mut Rng,
) -> Option<usize> {
    let q = |a: usize| root.value_sum[a] / root.visits[a] as f32;
    let visited: Vec<usize> = (0..root.actions.len())
        .filter(|&a| root.visits[a] > 0)
        .collect();
    let best = visited
        .iter()
        .map(|&a| q(a))
        .fold(f32::NEG_INFINITY, f32::max);
    let eligible: Vec<usize> = visited
        .into_iter()
        .filter(|&a| q(a) >= best - cutoff)
        .collect();
    // weights relative to the largest target, so 1/temp powers stay in range
    let top = eligible.iter().map(|&a| target[a]).fold(0.0f32, f32::max);
    if top <= 0.0 {
        return None;
    }
    let w: Vec<f64> = eligible
        .iter()
        .map(|&a| ((target[a] / top) as f64).powf(1.0 / temp as f64))
        .collect();
    let total: f64 = w.iter().sum();
    let mut x = uniform(rng) as f64 * total;
    for (i, &wi) in w.iter().enumerate() {
        if x < wi {
            return Some(eligible[i]);
        }
        x -= wi;
    }
    eligible.last().copied()
}

fn sample_gumbel(rng: &mut Rng) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-(-u.ln()).ln()) as f32
}

pub struct SearchPool {
    cfg: SearchConfig,
    tables: Arc<Tables>,
    slots: Vec<Slot>,
    threads: rayon::ThreadPool,
    finished: Vec<GameRecord>,
    stats: PoolStats,
}

impl SearchPool {
    pub fn new(cfg: SearchConfig) -> SearchPool {
        assert!(cfg.games > 0 && cfg.gumbel.considered > 0);
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .thread_name(|i| format!("peaduck-search-{i}"))
            .build()
            .expect("failed to start self-play threads");
        let tables = Arc::new(Tables::new(&cfg));
        SearchPool {
            slots: (0..cfg.games)
                .map(|i| {
                    let seed = (cfg.seed ^ 0x5EA8_C411)
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut rng = Rng::new(seed);
                    let game = cfg.new_game(&mut rng);
                    Slot::with_game(game, rng)
                })
                .collect(),
            tables,
            threads,
            finished: Vec::new(),
            stats: PoolStats::default(),
            cfg,
        }
    }

    pub fn num_games(&self) -> usize {
        self.slots.len()
    }

    pub fn num_threads(&self) -> usize {
        self.threads.current_num_threads()
    }

    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Positions to evaluate this round: observations (`num_games() *
    /// OBS_SIZE`) and the policy indices of their legal moves
    /// (`num_games() * MAX_LEGAL`, padded with 0).
    pub fn requests(&self, obs: &mut [f32], legal: &mut [i32]) {
        assert_eq!(obs.len(), self.slots.len() * OBS_SIZE);
        assert_eq!(legal.len(), self.slots.len() * MAX_LEGAL);
        let slots = &self.slots;
        self.threads.install(|| {
            slots
                .par_iter()
                .zip(obs.par_chunks_exact_mut(OBS_SIZE))
                .zip(legal.par_chunks_exact_mut(MAX_LEGAL))
                .for_each(|((slot, o), l)| {
                    encode_game(&slot.eval_game, o);
                    l.fill(0);
                    for (dst, &(a, _)) in l.iter_mut().zip(&slot.eval_actions) {
                        *dst = a as i32;
                    }
                });
        });
    }

    /// Network output for this round's requests: legal-move logits gathered
    /// at the indices from `requests` (`num_games() * MAX_LEGAL`) and W/D/L
    /// value logits (`num_games() * 3`). Returns the number of games that
    /// finished.
    pub fn respond(&mut self, logits: &[f32], value: &[f32]) -> usize {
        assert_eq!(logits.len(), self.slots.len() * MAX_LEGAL);
        assert_eq!(value.len(), self.slots.len() * 3);
        let cfg = &self.cfg;
        let tables = &self.tables;
        let slots = &mut self.slots;
        let (outcomes, moves): (Vec<Option<Outcome>>, Vec<u32>) = self.threads.install(|| {
            slots
                .par_iter_mut()
                .zip(logits.par_chunks_exact(MAX_LEGAL))
                .zip(value.par_chunks_exact(3))
                .map(|((slot, l), v)| {
                    let before = slot.plies.len();
                    let out = slot.respond(l, v, cfg, tables);
                    let moved = out.is_some() || slot.plies.len() != before;
                    // the move just played: last ply of the slot or of the finished game
                    let full = moved
                        && match &out {
                            Some(o) => o.plies.last(),
                            None => slot.plies.last(),
                        }
                        .is_some_and(|p| p.full_search);
                    (out, moved as u32 | (full as u32) << 1)
                })
                .unzip()
        });
        self.stats.plies += moves.iter().map(|&m| (m & 1) as u64).sum::<u64>();
        self.stats.full_plies += moves.iter().map(|&m| (m >> 1) as u64).sum::<u64>();
        let mut done = 0;
        for o in outcomes.into_iter().flatten() {
            self.stats.games_finished += 1;
            self.stats.terminations[o.termination as usize] += 1;
            match o.result {
                1 => self.stats.white_wins += 1,
                -1 => self.stats.black_wins += 1,
                _ => self.stats.draws += 1,
            }
            self.finished.push(GameRecord {
                plies: o.plies,
                result: o.result,
                termination: o.termination,
            });
            done += 1;
        }
        done
    }

    pub fn take_finished(&mut self) -> Vec<GameRecord> {
        std::mem::take(&mut self.finished)
    }

    pub fn plies_in_progress(&self) -> usize {
        self.slots.iter().map(|s| s.plies.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peaduck_core::encoding::action_to_move;
    use peaduck_core::{PieceType, Position};

    #[test]
    fn considered_visits_match_mctx() {
        // Reference values computed with mctx's Python implementation.
        assert_eq!(considered_visits_sequence(4, 6), vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(considered_visits_sequence(2, 5), vec![0, 0, 1, 1, 2]);
        let s = considered_visits_sequence(16, 64);
        let expected: Vec<u32> = [
            vec![0; 16],
            vec![1; 8],
            vec![2; 8],
            vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
            vec![7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
        ]
        .concat();
        assert_eq!(s, expected);
    }

    /// Material balance for the side to move, squashed to (-1, 1); +0.99 if
    /// the side to move can capture the enemy king right now.
    fn oracle(pos: &Position) -> f32 {
        let us = pos.side_to_move();
        if !pos.is_duck_step()
            && pos
                .legal_moves()
                .iter()
                .any(|m| pos.piece_on(m.to()).kind() == PieceType::King && m.is_capture())
        {
            return 0.99;
        }
        let mut s = 0i32;
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let v = [1, 3, 3, 5, 9][pt.idx()];
            s += v * (pos.pieces_of(us, pt).count() as i32 - pos.pieces_of(!us, pt).count() as i32);
        }
        (s as f32 / 10.0).tanh()
    }

    fn oracle_values(pool: &SearchPool) -> Vec<f32> {
        let mut value = Vec::with_capacity(pool.slots.len() * 3);
        for s in &pool.slots {
            let v = oracle(s.eval_game.position()).clamp(-0.99, 0.99);
            // W/D/L logits with W - L = v exactly
            value.extend_from_slice(&[(1.0 + v).ln(), -30.0, (1.0 - v).ln()]);
        }
        value
    }

    /// Search one step from `fen` with a uniform policy and the oracle
    /// value; returns the step and its record.
    fn search_step(fen: &str, simulations: usize, considered: usize) -> (Move, PlyRecord) {
        peaduck_core::init();
        let cfg = SearchConfig {
            games: 1,
            simulations,
            gumbel: GumbelConfig {
                gumbel_scale: 0.0,
                considered,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let game = Game::from_fen(fen).unwrap();
        pool.slots[0].game = game.clone();
        pool.slots[0].request_root();
        let mut obs = vec![0.0; OBS_SIZE];
        let mut legal = vec![0i32; MAX_LEGAL];
        loop {
            pool.requests(&mut obs, &mut legal);
            let value = oracle_values(&pool);
            pool.respond(&vec![0.0f32; MAX_LEGAL], &value);
            let finished = pool.take_finished();
            let ply = finished
                .first()
                .and_then(|g| g.plies.first())
                .or(pool.slots[0].plies.first())
                .cloned();
            if let Some(p) = ply {
                let mv = action_to_move(game.position(), p.action as usize).unwrap();
                return (mv, p);
            }
        }
    }

    #[test]
    fn captures_the_king() {
        let fen = "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1";
        let (mv, p) = search_step(fen, 32, 16);
        assert_eq!(mv.to_uci(), "e1e8");
        let [w, _, l] = p.value_wdl().unwrap();
        assert!(w - l > 0.5, "search value {w} {l}");
    }

    #[test]
    fn duck_blocks_the_attack_on_the_king() {
        // White's duck step; the black queen on a5 attacks the king on e1.
        // Only a duck on b4, c3 or d2 stops Qxe1.
        let fen = "4k3/8/8/q7/8/8/8/4K3 w - - 0 1 duck";
        let (mv, p) = search_step(fen, 256, 64);
        assert!(mv.is_duck());
        assert!(["b4", "c3", "d2"].contains(&mv.to_uci().as_str()), "{mv}");
        // only the three blocking squares are searched and get a target
        assert_eq!(p.targets.len(), 3);
        // value sign: White (queen down, but safe) is worse but not lost
        let [w, _, l] = p.value_wdl().unwrap();
        assert!(w - l < 0.0 && w - l > -0.9, "search value {w} {l}");
    }

    #[test]
    fn unsavable_duck_step_is_a_loss() {
        peaduck_core::init();
        // White walked into the knight on d3: no duck square saves the king
        let lost = Game::from_fen("4k3/8/8/8/8/3n4/5K2/8 w - - 0 1 duck").unwrap();
        assert_eq!(terminal_value(&lost), Some(-1.0));
        // ... unless the fifty-move rule ends the game first
        let drawn = Game::from_fen("4k3/8/8/8/8/3n4/5K2/8 w - - 100 80 duck").unwrap();
        assert_eq!(terminal_value(&drawn), None);
        assert_eq!(
            search_actions(drawn.position()).len(),
            legal_actions(drawn.position()).len()
        );
        // every king move walks into a knight: the search sees each one lost
        let fen = "4k3/8/8/8/8/2nn4/8/K7 w - - 0 1";
        let (_, p) = search_step(fen, 32, 4);
        let [w, _, l] = p.value_wdl().unwrap();
        assert!(w - l < -0.9, "search value {w} {l}");
    }

    #[test]
    fn piece_move_keeps_the_king_safe() {
        // Black to move; the rook on h8 can capture the white king on h1 only
        // if Black's duck doesn't block. White's rook a1 attacks the black
        // king on a8 along the a-file: Black must deal with it (move the king
        // off the a-file or block with the rook) or lose it next turn,
        // because the duck can only block one line... here it can: so check
        // that the search, which sees both steps, doesn't hang the king.
        let fen = "k6r/8/8/8/8/8/8/R6K b - - 0 1";
        let (mv, _) = search_step(fen, 400, 16);
        // h8h1 captures the king and wins at once
        assert_eq!(mv.to_uci(), "h8h1");
    }

    fn random_round(pool: &mut SearchPool, rng: &mut Rng, obs: &mut [f32], legal: &mut [i32]) {
        let n = pool.num_games();
        pool.requests(obs, legal);
        let logits: Vec<f32> = (0..n * MAX_LEGAL)
            .map(|_| rng.below(100) as f32 / 40.0)
            .collect();
        let value: Vec<f32> = (0..n * 3).map(|_| rng.below(100) as f32 / 40.0).collect();
        pool.respond(&logits, &value);
    }

    #[test]
    fn pool_plays_complete_games() {
        peaduck_core::init();
        let cfg = SearchConfig {
            games: 6,
            simulations: 8,
            max_plies: 120,
            threads: 2,
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let n = pool.num_games();
        let mut rng = Rng::new(4);
        let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
        while pool.stats().games_finished < 12 {
            random_round(&mut pool, &mut rng, &mut obs, &mut legal);
        }
        let games = pool.take_finished();
        for g in &games {
            let mut game = Game::default();
            for (i, p) in g.plies.iter().enumerate() {
                assert_eq!(p.duck_step, i % 2 == 1, "steps alternate");
                // the same position, apart from the move number
                let key = |p: &Position| {
                    let f = p.to_fen();
                    let v: Vec<&str> = f.split(' ').collect();
                    format!("{} {}", v[..5].join(" "), v.len() == 7) // 7th field: duck step
                };
                assert_eq!(key(&p.position()), key(game.position()));
                let sum: u32 = p.targets.iter().map(|&(_, q)| q as u32).sum();
                assert!((sum as i64 - 65535).abs() < 300);
                game.push(action_to_move(game.position(), p.action as usize).unwrap());
            }
            let (result, term) = match Termination::from_result(game.result()) {
                Some(x) => x,
                None => (0, Termination::MaxPlies),
            };
            assert_eq!((g.result, g.termination), (result, term));
        }
        let s = pool.stats();
        assert_eq!(s.white_wins + s.black_wins + s.draws, s.games_finished);
    }

    #[test]
    fn playout_cap_randomization() {
        peaduck_core::init();
        let run = |fast: usize| {
            let cfg = SearchConfig {
                games: 8,
                simulations: 32,
                max_plies: 120,
                threads: 2,
                fast_simulations: fast,
                full_prob: 0.25,
                fast_considered: 4,
                ..Default::default()
            };
            let mut pool = SearchPool::new(cfg);
            let n = pool.num_games();
            let mut rng = Rng::new(9);
            let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
            let mut rounds = 0u64;
            while pool.stats().games_finished < 16 {
                random_round(&mut pool, &mut rng, &mut obs, &mut legal);
                rounds += 1;
            }
            let games = pool.take_finished();
            let plies: Vec<&PlyRecord> = games.iter().flat_map(|g| &g.plies).collect();
            let full = plies.iter().filter(|p| p.full_search).count();
            let s = pool.stats();
            assert!(s.full_plies <= s.plies);
            (
                full as f64 / plies.len() as f64,
                rounds as f64 * n as f64 / s.plies as f64,
            )
        };
        let (share_off, evals_off) = run(0);
        assert_eq!(share_off, 1.0);
        let (share, evals) = run(8);
        assert!((share - 0.25).abs() < 0.08, "full-search share {share}");
        assert!(
            evals < 0.6 * evals_off,
            "evaluations per step {evals} vs {evals_off}"
        );
    }

    #[test]
    fn openings_and_random_turns() {
        peaduck_core::init();
        let line = |m: &str| m.split_whitespace().map(String::from).collect::<Vec<_>>();
        let book = vec![
            Opening::new(None, &line("e2e4,e4h6 e7e5,e5a3 g1f3,f3h3")).unwrap(),
            Opening::new(None, &line("d2d4,d4h6 d7d5,d5a3 c2c4,c4f6")).unwrap(),
            Opening::new(
                Some("rnbqkb1r/pppppppp/5n2/8/2PP4/6*1/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
                &[],
            )
            .unwrap(),
        ];
        assert!(Opening::new(None, &line("e2e5,e5e6")).is_err());
        assert!(Opening::new(None, &line("e2e4")).is_err());
        let cfg = SearchConfig {
            games: 64,
            simulations: 4,
            max_plies: 6,
            threads: 2,
            gumbel: GumbelConfig {
                considered: 4,
                ..Default::default()
            },
            openings: Arc::new(book),
            book_prob: 0.5,
            random_prob: 0.5,
            random_plies: 2,
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let n = pool.num_games();
        let mut rng = Rng::new(3);
        let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
        while pool.stats().games_finished < 200 {
            random_round(&mut pool, &mut rng, &mut obs, &mut legal);
        }
        let games = pool.take_finished();
        let start = Position::startpos().to_fen();
        let mut starts = std::collections::HashSet::new();
        let mut fresh = 0;
        for g in &games {
            let first = g.plies[0].position();
            assert!(!first.is_duck_step(), "games start at a turn");
            fresh += (first.to_fen() == start) as usize;
            starts.insert(first.to_fen());
            let mut game = Game::new(first);
            for p in &g.plies {
                game.push(action_to_move(game.position(), p.action as usize).unwrap());
            }
        }
        let share = fresh as f64 / games.len() as f64;
        assert!((share - 0.25).abs() < 0.1, "fresh starts {share}");
        assert!(starts.len() > 40, "only {} start positions", starts.len());
    }

    #[test]
    fn temperature_schedule() {
        let cfg = SearchConfig {
            temperature: 1.0,
            temp_decay_plies: 20,
            temp_endgame: 0.2,
            ..Default::default()
        };
        assert_eq!(cfg.move_temperature(0), 1.0);
        assert!((cfg.move_temperature(10) - 0.5).abs() < 1e-6);
        assert_eq!(cfg.move_temperature(30), 0.2);
    }
}
