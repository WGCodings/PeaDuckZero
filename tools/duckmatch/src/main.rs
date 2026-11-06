//! duckmatch — plays duck chess games between UCI engines and tests the
//! result with an SPRT (pentanomial, like OpenBench / fastchess, which don't
//! support duck chess). Moves are whole turns in Fairy-Stockfish notation
//! (`e2e4,e4d5`); every move is checked with PeaDuckZero's move generator.
//!
//! ```text
//! duckmatch --engine cmd=./dev name=dev --engine cmd=./base name=base \
//!           --each tc=8+0.08 option.Hash=16 option.Threads=1 \
//!           --games 2000 --concurrency 8 --sprt elo0=0 elo1=5
//! ```
//!
//! Options:
//! * `--engine k=v ...`: one engine (twice). Keys: `cmd` (path), `name`,
//!   `arg` (repeatable), `tc`, `nodes`, `movetime`, `option.<Name>=<value>`.
//! * `--each k=v ...`: keys for both engines.
//! * `tc=BASE+INC` (seconds, e.g. `8+0.08`), `nodes=N` (`go nodes N`) or
//!   `movetime=MS`.
//! * `--games N`: games in all (rounded up to pairs: every opening is played
//!   twice, colours swapped). Default 100.
//! * `--concurrency K`: games at the same time. Default 1.
//! * `--openings FILE`: one opening per line, `startpos moves ...`,
//!   `fen <FEN> [moves ...]` or a bare FEN (duck as `*`), used in order and
//!   repeated. Default: `--random-turns` random turns from the start position.
//! * `--random-turns N`: random turns per opening (default 4); turns that let
//!   the opponent capture the king at once are avoided.
//! * `--sprt elo0=A elo1=B [alpha=0.05] [beta=0.05]`: stop when the LLR
//!   crosses a bound (logistic Elo, pentanomial normal approximation).
//! * `--max-turns N`: adjudicate a draw after N turns (default 400).
//! * `--margin MS`: time a side may overstep its clock (default 100).
//! * `--log FILE`: one line per game (result, termination, opening, moves).
//! * `--seed S`: opening seed (default 0).
//! * `--json`: also print the summary as one JSON line, every 10 games
//!   (`"final": false`) and at the end (`"final": true`).

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use peaduck_core::encoding::legal_actions;
use peaduck_core::rng::Rng;
use peaduck_core::{Color, Game, GameResult, Position};

// ------------------------------------------------------------------ setup

#[derive(Clone, Debug, Default)]
struct EngineSpec {
    name: String,
    cmd: String,
    args: Vec<String>,
    options: Vec<(String, String)>,
    limit: Limit,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Limit {
    #[default]
    Unset,
    /// base and increment in milliseconds
    Clock(u64, u64),
    Nodes(u64),
    MoveTime(u64),
}

#[derive(Clone, Copy, Debug)]
struct Sprt {
    elo0: f64,
    elo1: f64,
    alpha: f64,
    beta: f64,
}

struct Config {
    engines: Vec<EngineSpec>,
    games: usize,
    concurrency: usize,
    openings: Vec<Opening>,
    sprt: Option<Sprt>,
    max_turns: usize,
    margin_ms: u64,
    log: Option<String>,
    json: bool,
}

#[derive(Clone, Debug)]
struct Opening {
    /// None = start position
    fen: Option<String>,
    moves: Vec<String>,
}

impl Opening {
    fn parse(line: &str) -> Result<Opening, String> {
        let line = line.trim();
        let (setup, moves) = match line.split_once(" moves ") {
            Some((a, b)) => (a.trim(), b.split_whitespace().map(String::from).collect()),
            None => (line.strip_suffix(" moves").unwrap_or(line), Vec::new()),
        };
        let fen = if setup == "startpos" {
            None
        } else {
            Some(
                setup
                    .strip_prefix("fen ")
                    .unwrap_or(setup)
                    .trim()
                    .to_string(),
            )
        };
        let o = Opening { fen, moves };
        o.game()?;
        Ok(o)
    }

    fn game(&self) -> Result<Game, String> {
        let mut g = match &self.fen {
            Some(f) => Game::from_fen(f).map_err(|e| e.to_string())?,
            None => Game::default(),
        };
        for m in &self.moves {
            if !g.push_uci(m) {
                return Err(format!("illegal move {m} in opening"));
            }
        }
        if g.position().is_duck_step() || g.result().is_over() {
            return Err("opening must end at the start of a turn of an ongoing game".into());
        }
        Ok(g)
    }

    fn position_cmd(&self, extra: &[String]) -> String {
        let mut s = match &self.fen {
            Some(f) => format!("position fen {f}"),
            None => "position startpos".to_string(),
        };
        if !self.moves.is_empty() || !extra.is_empty() {
            s.push_str(" moves");
            for m in self.moves.iter().chain(extra) {
                s.push(' ');
                s.push_str(m);
            }
        }
        s
    }

    fn describe(&self) -> String {
        self.position_cmd(&[])
            .trim_start_matches("position ")
            .to_string()
    }
}

/// Random openings: `turns` random turns from the start position, avoiding
/// turns after which the opponent can capture the king at once.
fn random_openings(n: usize, turns: usize, seed: u64) -> Vec<Opening> {
    let mut rng = Rng::new(seed ^ 0xD0C4_0FE7);
    let mut out = Vec::with_capacity(n);
    'outer: while out.len() < n {
        let mut g = Game::default();
        let mut moves = Vec::new();
        for _ in 0..turns {
            let mut options = Vec::new();
            for (_, m) in legal_actions(g.position()) {
                let mut after = *g.position();
                after.make_move(m);
                for (_, d) in legal_actions(&after) {
                    let mut end = after;
                    end.make_move(d);
                    if !end.king_capture_available() {
                        options.push((m, d));
                    }
                }
            }
            if options.is_empty() {
                continue 'outer;
            }
            let (m, d) = options[rng.below(options.len() as u64) as usize];
            moves.push(peaduck_core::moves::turn_uci(m, Some(d.to())));
            g.push(m);
            g.push(d);
            if g.result().is_over() {
                continue 'outer;
            }
        }
        out.push(Opening { fen: None, moves });
    }
    out
}

fn parse_tc(s: &str) -> Result<Limit, String> {
    let (base, inc) = s.split_once('+').unwrap_or((s, "0"));
    let secs = |x: &str| {
        x.parse::<f64>()
            .map_err(|_| format!("bad time control {s:?} (want BASE+INC seconds)"))
    };
    Ok(Limit::Clock(
        (secs(base)? * 1000.0).round() as u64,
        (secs(inc)? * 1000.0).round() as u64,
    ))
}

fn apply_kv(e: &mut EngineSpec, kv: &str) -> Result<(), String> {
    let (k, v) = kv
        .split_once('=')
        .ok_or_else(|| format!("expected key=value, got {kv:?}"))?;
    match k {
        "cmd" => e.cmd = v.to_string(),
        "name" => e.name = v.to_string(),
        "arg" => e.args.push(v.to_string()),
        "tc" => e.limit = parse_tc(v)?,
        "nodes" => e.limit = Limit::Nodes(v.parse().map_err(|_| format!("bad nodes {v:?}"))?),
        "movetime" => {
            e.limit = Limit::MoveTime(v.parse().map_err(|_| format!("bad movetime {v:?}"))?)
        }
        _ => match k.strip_prefix("option.") {
            Some(name) => e.options.push((name.to_string(), v.to_string())),
            None => return Err(format!("unknown engine key {k:?}")),
        },
    }
    Ok(())
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut engines: Vec<Vec<String>> = Vec::new();
    let mut each: Vec<String> = Vec::new();
    let mut games: usize = 100;
    let mut concurrency: usize = 1;
    let mut openings_file = None;
    let mut random_turns: usize = 4;
    let mut sprt = None;
    let mut max_turns: usize = 400;
    let mut margin_ms: u64 = 100;
    let mut log = None;
    let mut seed = 0u64;
    let mut json = false;
    let mut i = 0;
    let value = |i: usize| -> Result<&String, String> {
        args.get(i + 1)
            .ok_or_else(|| format!("{} needs a value", args[i]))
    };
    while i < args.len() {
        let a = args[i].as_str();
        // the k=v words following an option
        let mut kvs = || {
            let mut v = Vec::new();
            while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                i += 1;
                v.push(args[i].clone());
            }
            v
        };
        match a {
            "--engine" => engines.push(kvs()),
            "--each" => each.extend(kvs()),
            "--sprt" => {
                let mut s = Sprt {
                    elo0: 0.0,
                    elo1: 5.0,
                    alpha: 0.05,
                    beta: 0.05,
                };
                for kv in kvs() {
                    let (k, v) = kv.split_once('=').ok_or(format!("bad --sprt {kv:?}"))?;
                    let x: f64 = v.parse().map_err(|_| format!("bad --sprt {kv:?}"))?;
                    match k {
                        "elo0" => s.elo0 = x,
                        "elo1" => s.elo1 = x,
                        "alpha" => s.alpha = x,
                        "beta" => s.beta = x,
                        _ => return Err(format!("bad --sprt key {k:?}")),
                    }
                }
                sprt = Some(s);
            }
            "--games" => {
                games = value(i)?.parse().map_err(|_| "bad --games")?;
                i += 1;
            }
            "--concurrency" => {
                concurrency = value(i)?.parse().map_err(|_| "bad --concurrency")?;
                i += 1;
            }
            "--openings" => {
                openings_file = Some(value(i)?.clone());
                i += 1;
            }
            "--random-turns" => {
                random_turns = value(i)?.parse().map_err(|_| "bad --random-turns")?;
                i += 1;
            }
            "--max-turns" => {
                max_turns = value(i)?.parse().map_err(|_| "bad --max-turns")?;
                i += 1;
            }
            "--margin" => {
                margin_ms = value(i)?.parse().map_err(|_| "bad --margin")?;
                i += 1;
            }
            "--log" => {
                log = Some(value(i)?.clone());
                i += 1;
            }
            "--seed" => {
                seed = value(i)?.parse().map_err(|_| "bad --seed")?;
                i += 1;
            }
            "--json" => json = true,
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument {a:?}")),
        }
        i += 1;
    }
    if engines.len() != 2 {
        return Err("need exactly two --engine".into());
    }
    let mut specs = Vec::new();
    for (n, kvs) in engines.iter().enumerate() {
        let mut e = EngineSpec::default();
        for kv in each.iter().chain(kvs) {
            apply_kv(&mut e, kv)?;
        }
        if e.cmd.is_empty() {
            return Err(format!("engine {} has no cmd=", n + 1));
        }
        if e.name.is_empty() {
            e.name = format!("engine{}", n + 1);
        }
        if e.limit == Limit::Unset {
            return Err(format!("engine {} has no tc=, nodes= or movetime=", e.name));
        }
        specs.push(e);
    }
    if specs[0].name == specs[1].name {
        specs[1].name.push_str("-2");
    }
    let pairs = games.div_ceil(2).max(1);
    let openings = match openings_file {
        Some(path) => {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
            let list: Vec<Opening> = text
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .enumerate()
                .map(|(n, l)| Opening::parse(l).map_err(|e| format!("{path} line {}: {e}", n + 1)))
                .collect::<Result<_, _>>()?;
            if list.is_empty() {
                return Err(format!("{path}: no openings"));
            }
            (0..pairs).map(|k| list[k % list.len()].clone()).collect()
        }
        None => random_openings(pairs, random_turns, seed),
    };
    Ok(Config {
        engines: specs,
        games: pairs * 2,
        concurrency: concurrency.max(1),
        openings,
        sprt,
        max_turns,
        margin_ms,
        log,
        json,
    })
}

// ------------------------------------------------------------------ engines

struct Engine {
    name: String,
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    fn start(spec: &EngineSpec) -> Result<Engine, String> {
        let mut child = Command::new(&spec.cmd)
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", spec.cmd))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut e = Engine {
            name: spec.name.clone(),
            child,
            stdin,
            lines: rx,
        };
        e.send("uci")?;
        e.wait_for("uciok", Duration::from_secs(30))?;
        for (k, v) in &spec.options {
            e.send(&format!("setoption name {k} value {v}"))?;
        }
        e.send("setoption name UCI_Variant value duck")?;
        e.ready()?;
        Ok(e)
    }

    fn send(&mut self, s: &str) -> Result<(), String> {
        writeln!(self.stdin, "{s}")
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("{}: write failed ({e})", self.name))
    }

    fn wait_for(&mut self, prefix: &str, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(l) if l.starts_with(prefix) => return Ok(l),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!("{}: no '{prefix}' in time", self.name))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(format!("{}: engine exited", self.name))
                }
            }
        }
    }

    fn ready(&mut self) -> Result<(), String> {
        self.send("isready")?;
        self.wait_for("readyok", Duration::from_secs(60))
            .map(|_| ())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ------------------------------------------------------------------ games

#[derive(Clone, Debug)]
struct GameOutcome {
    pair: usize,
    /// engine index playing White
    white: usize,
    /// score of engine 0 (1, 0.5, 0)
    score0: f64,
    termination: String,
    turns: Vec<String>,
    result: &'static str,
}

/// Play one game; `engines[white]` has White. Errors (crash, timeout,
/// illegal move) lose for the side that caused them.
fn play_game(
    cfg: &Config,
    engines: &mut [Option<Engine>; 2],
    opening: &Opening,
    white: usize,
) -> (f64, String, Vec<String>) {
    let mut game = opening.game().expect("checked opening");
    let mut turns: Vec<String> = Vec::new();
    let mut clock = [0i64; 2];
    for (i, c) in clock.iter_mut().enumerate() {
        if let Limit::Clock(base, _) = cfg.engines[i].limit {
            *c = base as i64;
        }
    }
    for e in engines.iter_mut().flatten() {
        let _ = e.send("ucinewgame");
        let _ = e.ready();
    }
    // score for engine 0 when engine `loser` loses
    let lose = |loser: usize| if loser == 0 { 0.0 } else { 1.0 };
    loop {
        match game.result() {
            GameResult::Ongoing => {}
            GameResult::Win(c, why) => {
                let winner_idx = if c == Color::White { white } else { 1 - white };
                return (lose(1 - winner_idx), format!("{why:?}"), turns);
            }
            GameResult::Draw(why) => return (0.5, format!("{why:?}"), turns),
        }
        if turns.len() >= cfg.max_turns {
            return (0.5, "MaxTurns".into(), turns);
        }
        let stm = game.position().side_to_move();
        let idx = if stm == Color::White {
            white
        } else {
            1 - white
        };
        let Some(e) = engines[idx].as_mut() else {
            return (lose(idx), "EngineError".into(), turns);
        };
        let go = match cfg.engines[idx].limit {
            Limit::Clock(_, _) => {
                let (w, b) = if white == 0 {
                    (clock[0], clock[1])
                } else {
                    (clock[1], clock[0])
                };
                let inc = |i: usize| match cfg.engines[i].limit {
                    Limit::Clock(_, inc) => inc,
                    _ => 0,
                };
                let (wi, bi) = if white == 0 {
                    (inc(0), inc(1))
                } else {
                    (inc(1), inc(0))
                };
                format!(
                    "go wtime {} btime {} winc {wi} binc {bi}",
                    w.max(1),
                    b.max(1)
                )
            }
            Limit::Nodes(n) => format!("go nodes {n}"),
            Limit::MoveTime(ms) => format!("go movetime {ms}"),
            Limit::Unset => unreachable!(),
        };
        let pos_cmd = opening.position_cmd(&turns);
        let sent = e.send(&pos_cmd).and_then(|_| e.send(&go));
        let start = Instant::now();
        let timeout = match cfg.engines[idx].limit {
            Limit::Clock(_, _) => Duration::from_millis(clock[idx].max(0) as u64 + cfg.margin_ms),
            Limit::MoveTime(ms) => Duration::from_millis(ms + cfg.margin_ms + 1000),
            _ => Duration::from_secs(600),
        };
        let reply = sent.and_then(|_| e.wait_for("bestmove", timeout));
        let elapsed = start.elapsed().as_millis() as i64;
        let line = match reply {
            Ok(l) => l,
            Err(err) => {
                let timeout = err.contains("in time");
                engines[idx] = None; // restarted for the next game
                return (
                    lose(idx),
                    if timeout {
                        "TimeForfeit"
                    } else {
                        "EngineError"
                    }
                    .into(),
                    turns,
                );
            }
        };
        if let Limit::Clock(_, inc) = cfg.engines[idx].limit {
            clock[idx] -= elapsed;
            if clock[idx] < -(cfg.margin_ms as i64) {
                return (lose(idx), "TimeForfeit".into(), turns);
            }
            clock[idx] = clock[idx].max(0) + inc as i64;
        }
        let mv = line.split_whitespace().nth(1).unwrap_or("").to_string();
        if !game.push_uci(&mv) {
            return (lose(idx), format!("IllegalMove({mv})"), turns);
        }
        turns.push(mv);
    }
}

fn worker(
    cfg: Arc<Config>,
    jobs: Arc<Mutex<Vec<(usize, usize)>>>,
    results: mpsc::Sender<GameOutcome>,
    stop: Arc<AtomicBool>,
) {
    let mut engines: [Option<Engine>; 2] = [None, None];
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Some((pair, white)) = jobs.lock().unwrap().pop() else {
            return;
        };
        for (i, e) in engines.iter_mut().enumerate() {
            if e.is_none() {
                match Engine::start(&cfg.engines[i]) {
                    Ok(x) => *e = Some(x),
                    Err(err) => eprintln!("{err}"),
                }
            }
        }
        let opening = &cfg.openings[pair];
        let (score0, termination, turns) = play_game(&cfg, &mut engines, opening, white);
        // score of the White player decides the PGN result
        let white_score = if white == 0 { score0 } else { 1.0 - score0 };
        let result = if white_score > 0.75 {
            "1-0"
        } else if white_score < 0.25 {
            "0-1"
        } else {
            "1/2-1/2"
        };
        let out = GameOutcome {
            pair,
            white,
            score0,
            termination,
            turns,
            result,
        };
        if results.send(out).is_err() {
            return;
        }
    }
}

// ------------------------------------------------------------------ statistics

fn score_to_elo(s: f64) -> f64 {
    let s = s.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

fn elo_to_score(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

struct Stats {
    wins: usize,
    draws: usize,
    losses: usize,
    /// pairs scoring 0, 0.5, 1, 1.5, 2 for engine 0
    penta: [usize; 5],
}

impl Stats {
    fn pairs(&self) -> usize {
        self.penta.iter().sum()
    }

    /// Mean and variance of the pair score (0..1 per game).
    fn pair_mean_var(&self) -> Option<(f64, f64)> {
        let n = self.pairs() as f64;
        if n < 1.0 {
            return None;
        }
        let mean = (0..5)
            .map(|k| self.penta[k] as f64 * k as f64 / 4.0)
            .sum::<f64>()
            / n;
        let var = (0..5)
            .map(|k| self.penta[k] as f64 * (k as f64 / 4.0 - mean).powi(2))
            .sum::<f64>()
            / n;
        Some((mean, var))
    }

    /// Elo of engine 0 with a 95% interval (pentanomial).
    fn elo(&self) -> Option<(f64, f64)> {
        let (mean, var) = self.pair_mean_var()?;
        let n = self.pairs() as f64;
        let se = (var / n).sqrt().max(0.25 / n);
        let elo = score_to_elo(mean);
        let hi = score_to_elo(mean + 1.96 * se);
        let lo = score_to_elo(mean - 1.96 * se);
        Some((elo, (hi - lo) / 2.0))
    }

    /// Log-likelihood ratio of elo1 vs elo0 (normal approximation on the
    /// pair scores, as fishtest's pentanomial SPRT).
    fn llr(&self, s: &Sprt) -> f64 {
        let Some((mean, var)) = self.pair_mean_var() else {
            return 0.0;
        };
        if var <= 0.0 {
            return 0.0;
        }
        let (s0, s1) = (elo_to_score(s.elo0), elo_to_score(s.elo1));
        self.pairs() as f64 * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)
    }
}

fn bounds(s: &Sprt) -> (f64, f64) {
    (
        (s.beta / (1.0 - s.alpha)).ln(),
        ((1.0 - s.beta) / s.alpha).ln(),
    )
}

fn summary(cfg: &Config, st: &Stats, terms: &BTreeMap<String, usize>) -> String {
    let games = st.wins + st.draws + st.losses;
    let mut s = format!(
        "{} vs {}: games {games} (+{} ={} -{}) penta {:?}",
        cfg.engines[0].name, cfg.engines[1].name, st.wins, st.draws, st.losses, st.penta
    );
    if let Some((elo, err)) = st.elo() {
        s.push_str(&format!(" | Elo {elo:+.1} +- {err:.1}"));
    }
    if let Some(sp) = &cfg.sprt {
        let (lo, hi) = bounds(sp);
        s.push_str(&format!(
            " | LLR {:.2} ({lo:.2}, {hi:.2}) [{}, {}]",
            st.llr(sp),
            sp.elo0,
            sp.elo1
        ));
    }
    let t: Vec<String> = terms.iter().map(|(k, v)| format!("{k} {v}")).collect();
    s.push_str(&format!(" | {}", t.join(", ")));
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse_args(&args) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            let doc: String = include_str!("main.rs")
                .lines()
                .take_while(|l| l.starts_with("//!"))
                .map(|l| {
                    l.trim_start_matches("//!")
                        .strip_prefix(' ')
                        .unwrap_or("")
                        .to_string()
                        + "\n"
                })
                .collect();
            eprint!("{doc}");
            std::process::exit(2);
        }
    };
    peaduck_core::init();
    let _ = Position::startpos();

    // jobs popped from the end: first pair first, both colours of a pair
    // close together
    let mut jobs: Vec<(usize, usize)> = (0..cfg.games / 2).flat_map(|p| [(p, 0), (p, 1)]).collect();
    jobs.reverse();
    let jobs = Arc::new(Mutex::new(jobs));
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    for _ in 0..cfg.concurrency.min(cfg.games) {
        let (cfg, jobs, tx, stop) = (cfg.clone(), jobs.clone(), tx.clone(), stop.clone());
        handles.push(std::thread::spawn(move || worker(cfg, jobs, tx, stop)));
    }
    drop(tx);

    let mut log = cfg
        .log
        .as_ref()
        .map(|p| File::create(p).expect("cannot create --log file"));
    let mut st = Stats {
        wins: 0,
        draws: 0,
        losses: 0,
        penta: [0; 5],
    };
    let mut terms: BTreeMap<String, usize> = BTreeMap::new();
    let mut half: HashMap<usize, f64> = HashMap::new();
    let mut done = 0;
    let mut total_turns = 0usize;
    let start = Instant::now();
    let mut decided = None;
    for g in rx {
        done += 1;
        total_turns += g.turns.len();
        if g.score0 > 0.75 {
            st.wins += 1;
        } else if g.score0 < 0.25 {
            st.losses += 1;
        } else {
            st.draws += 1;
        }
        *terms.entry(g.termination.clone()).or_default() += 1;
        if let Some(prev) = half.remove(&g.pair) {
            st.penta[((prev + g.score0) * 2.0).round() as usize] += 1;
        } else {
            half.insert(g.pair, g.score0);
        }
        if let Some(f) = log.as_mut() {
            let _ = writeln!(
                f,
                "{} white={} {} [{}] {}",
                g.result,
                cfg.engines[g.white].name,
                g.termination,
                cfg.openings[g.pair].describe(),
                g.turns.join(" ")
            );
        }
        if done % 10 == 0 || done == cfg.games {
            println!(
                "[{done}/{}, {:.0}s] {}",
                cfg.games,
                start.elapsed().as_secs_f64(),
                summary(&cfg, &st, &terms)
            );
            if cfg.json {
                println!("{}", json_line(&cfg, &st, &terms, None, total_turns, done, false));
            }
        }
        if let Some(sp) = &cfg.sprt {
            let (lo, hi) = bounds(sp);
            let llr = st.llr(sp);
            if decided.is_none() && (llr <= lo || llr >= hi) {
                decided = Some(if llr >= hi {
                    "H1 accepted"
                } else {
                    "H0 accepted"
                });
                stop.store(true, Ordering::Relaxed);
            }
        }
    }
    for h in handles {
        let _ = h.join();
    }
    println!("final: {}", summary(&cfg, &st, &terms));
    if let Some(d) = decided {
        println!("SPRT: {d}");
    }
    if cfg.json {
        println!("{}", json_line(&cfg, &st, &terms, decided, total_turns, done, true));
    }
}

/// The result as one JSON line (`final`: false for the progress lines that
/// `--json` also prints every 10 games).
fn json_line(
    cfg: &Config,
    st: &Stats,
    terms: &BTreeMap<String, usize>,
    decided: Option<&str>,
    total_turns: usize,
    done: usize,
    last: bool,
) -> String {
    let (elo, err) = st.elo().unwrap_or((0.0, 0.0));
    let llr = cfg.sprt.as_ref().map_or(0.0, |s| st.llr(s));
    let t: Vec<String> = terms.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    format!(
        "{{\"games\": {}, \"wins\": {}, \"draws\": {}, \"losses\": {}, \"pentanomial\": {:?}, \"elo\": {elo:.2}, \"elo_error\": {err:.2}, \"llr\": {llr:.3}, \"sprt\": {}, \"avg_turns\": {:.1}, \"final\": {last}, \"terminations\": {{{}}}}}",
        st.wins + st.draws + st.losses,
        st.wins,
        st.draws,
        st.losses,
        st.penta,
        decided.map_or("null".to_string(), |d| format!("\"{d}\"")),
        total_turns as f64 / done.max(1) as f64,
        t.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llr_and_elo() {
        let even = Stats {
            wins: 0,
            draws: 0,
            losses: 0,
            penta: [10, 40, 100, 40, 10],
        };
        let (elo, err) = even.elo().unwrap();
        assert!(elo.abs() < 1e-6 && err > 0.0);
        let s = Sprt {
            elo0: 0.0,
            elo1: 5.0,
            alpha: 0.05,
            beta: 0.05,
        };
        assert!(even.llr(&s) < 0.0);
        let strong = Stats {
            penta: [5, 20, 100, 60, 25],
            ..even
        };
        assert!(strong.llr(&s) > 2.0, "{}", strong.llr(&s));
        assert!(strong.elo().unwrap().0 > 30.0);
        let (lo, hi) = bounds(&s);
        assert!((lo + 2.944).abs() < 1e-3 && (hi - 2.944).abs() < 1e-3);
    }

    #[test]
    fn openings_parse_and_random_ones_are_safe() {
        peaduck_core::init();
        let o = Opening::parse("startpos moves e2e4,e4d5 e7e5,e5e6").unwrap();
        assert_eq!(o.moves.len(), 2);
        assert_eq!(
            o.position_cmd(&["g1f3,f3h3".into()]),
            "position startpos moves e2e4,e4d5 e7e5,e5e6 g1f3,f3h3"
        );
        let o =
            Opening::parse("rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert!(o.fen.is_some());
        assert!(Opening::parse("startpos moves e2e5,e5e6").is_err());
        for o in random_openings(20, 4, 1) {
            let g = o.game().unwrap();
            assert_eq!(o.moves.len(), 4);
            assert!(!g.position().king_capture_available());
        }
    }

    #[test]
    fn tc_parsing() {
        assert_eq!(parse_tc("8+0.08").unwrap(), Limit::Clock(8000, 80));
        assert_eq!(parse_tc("60").unwrap(), Limit::Clock(60000, 0));
        assert!(parse_tc("x+1").is_err());
    }
}
