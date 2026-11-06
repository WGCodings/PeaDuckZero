//! The network engine: owns the policy/value network and hands each `go` to
//! the search method selected with the UCI option `SearchMode`
//! (see search/mod.rs): `Mcts` (default; PUCT with policy priors and network
//! values) or `Searchless` (one evaluation per step, best policy step).

use std::sync::atomic::AtomicBool;

use peaduck_core::encoding::{encode_game, legal_actions, NUM_ACTIONS, OBS_SIZE};
use peaduck_core::{Game, Move};
use peaduck_nn::fast::{FastNet, FastWeights};
use peaduck_nn::wdl_probs;

use crate::engine::{CommonOptions, Engine, SearchLimits, SearchResult};
use crate::search::{params, NetSearch, SearchContext, SearchMode, SearchParams};

/// The network compiled into the binary (see build.rs / `EVALFILE`).
pub static EMBEDDED_NET: &[u8] = include_bytes!(env!("PEADUCK_EVALFILE"));

/// Result of evaluating one position with the policy/value heads.
pub struct Evaluation {
    /// Legal steps (piece moves, or duck moves in the duck step) with their
    /// raw policy logits.
    pub moves: Vec<(Move, f32)>,
    /// Win / draw / loss probabilities for the side to move.
    pub wdl: [f32; 3],
}

pub struct PolicyNet {
    net: FastNet,
    obs: Vec<f32>,
    policy: Vec<f32>,
}

impl PolicyNet {
    pub fn from_pdn(bytes: &[u8]) -> Result<PolicyNet, String> {
        let net = FastNet::new(std::sync::Arc::new(FastWeights::from_pdn(bytes)?));
        Ok(PolicyNet {
            net,
            obs: vec![0.0; OBS_SIZE],
            policy: vec![0.0; NUM_ACTIONS],
        })
    }

    /// Another instance of the same network for a search thread: the
    /// weights are shared, only the scratch buffers are new.
    pub fn fork(&self) -> PolicyNet {
        let mut net = FastNet::new(std::sync::Arc::clone(self.net.weights()));
        net.set_int8(self.net.int8());
        net.set_simd(self.net.simd());
        net.set_winograd(self.net.winograd());
        PolicyNet {
            net,
            obs: vec![0.0; OBS_SIZE],
            policy: vec![0.0; NUM_ACTIONS],
        }
    }

    /// Which inference kernels are in use (for `info string`).
    pub fn backend_name(&self) -> String {
        self.net.describe()
    }

    /// Network size, e.g. "64x6".
    pub fn size_name(&self) -> String {
        let w = self.net.weights();
        format!("{}x{}", w.channels(), w.blocks())
    }

    pub fn embedded() -> PolicyNet {
        PolicyNet::from_pdn(EMBEDDED_NET).expect("embedded network is invalid")
    }

    /// Evaluate the current position of `game`. Returns no moves if the
    /// game is over.
    pub fn evaluate(&mut self, game: &Game) -> Evaluation {
        let legal = legal_actions(game.position());
        encode_game(game, &mut self.obs);
        let mut value = [0.0f32; 3];
        self.net.evaluate(&self.obs, &mut self.policy, &mut value);
        Evaluation {
            moves: legal.iter().map(|&(a, m)| (m, self.policy[a])).collect(),
            wdl: wdl_probs(&value),
        }
    }
}

/// Map an expected score `q = W - L` in [-1, 1] to centipawns
/// (tangent curve: q 0.5 ≈ 111 cp, q 0.9 ≈ 671 cp, ±0.999 ≈ ±10800 cp).
pub fn q_to_cp(q: f32) -> i32 {
    let q = q.clamp(-0.999, 0.999) as f64;
    (111.714_640_912 * (1.562_068_842_1 * q).tan()).round() as i32
}

/// ` wdl W D L` in permille, as used in `info` lines.
pub fn wdl_string([w, d, _]: [f32; 3]) -> String {
    let permille = |x: f32| (x * 1000.0).round() as i32;
    format!(
        " wdl {} {} {}",
        permille(w),
        permille(d),
        1000 - permille(w) - permille(d)
    )
}

/// Legal moves with their policy probability (softmax over the legal
/// moves' logits), most likely first: (move, probability, raw logit).
pub fn move_probabilities(moves: &[(Move, f32)]) -> Vec<(Move, f32, f32)> {
    let max = moves.iter().map(|m| m.1).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = moves.iter().map(|m| (m.1 - max).exp()).sum();
    let mut out: Vec<(Move, f32, f32)> = moves
        .iter()
        .map(|&(m, logit)| (m, (logit - max).exp() / sum, logit))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

pub struct PolicyEngine {
    net: PolicyNet,
    net_name: String,
    show_wdl: bool,
    verbose_move_stats: bool,
    mode: SearchMode,
    search: Box<dyn NetSearch>,
    params: SearchParams,
}

impl PolicyEngine {
    pub const DEFAULT_MODE: SearchMode = SearchMode::Mcts;

    pub fn new() -> PolicyEngine {
        PolicyEngine {
            net: PolicyNet::embedded(),
            net_name: "<embedded>".to_string(),
            show_wdl: true,
            verbose_move_stats: false,
            mode: Self::DEFAULT_MODE,
            search: Self::DEFAULT_MODE.create(),
            params: SearchParams::default(),
        }
    }
}

impl Engine for PolicyEngine {
    fn new_game(&mut self) {
        self.search.new_game();
    }

    fn uci_options(&self) -> Vec<String> {
        let mut options = vec![
            format!("option name EvalFile type string default {}", self.net_name),
            "option name UCI_ShowWDL type check default true".to_string(),
            "option name VerboseMoveStats type check default false".to_string(),
            "option name UCI_Variant type combo default duck var duck".to_string(),
            SearchMode::uci_option(Self::DEFAULT_MODE),
        ];
        options.extend(params::uci_options());
        options
    }

    fn spsa_lines(&self) -> Vec<String> {
        params::spsa_lines(&self.params)
    }

    fn set_option(&mut self, name: &str, value: &str) -> bool {
        match name.to_ascii_lowercase().as_str() {
            "evalfile" => {
                if value.is_empty() || value == "<embedded>" {
                    self.net = PolicyNet::embedded();
                    self.net_name = "<embedded>".into();
                } else {
                    match std::fs::read(value)
                        .map_err(|e| e.to_string())
                        .and_then(|b| PolicyNet::from_pdn(&b))
                    {
                        Ok(net) => {
                            println!(
                                "info string loaded network {value} ({}, {} inference)",
                                net.size_name(),
                                net.backend_name()
                            );
                            self.net = net;
                            self.net_name = value.to_string();
                        }
                        Err(e) => println!("info string failed to load {value}: {e}"),
                    }
                }
                // cached evaluations belong to the old network
                self.search.new_game();
                true
            }
            "uci_variant" => {
                if !value.eq_ignore_ascii_case("duck") {
                    println!("info string only duck chess is supported (UCI_Variant duck)");
                }
                true
            }
            "uci_showwdl" => {
                self.show_wdl = value.eq_ignore_ascii_case("true");
                true
            }
            "verbosemovestats" => {
                self.verbose_move_stats = value.eq_ignore_ascii_case("true");
                true
            }
            "searchmode" => {
                match SearchMode::parse(value) {
                    Some(mode) if mode != self.mode => {
                        self.mode = mode;
                        self.search = mode.create();
                    }
                    Some(_) => {}
                    None => println!("info string unknown SearchMode {value}"),
                }
                true
            }
            _ => self.params.set(name, value),
        }
    }

    fn search(
        &mut self,
        game: &Game,
        limits: &SearchLimits,
        common: &CommonOptions,
        stop: &AtomicBool,
    ) -> SearchResult {
        let ctx = SearchContext {
            limits,
            common,
            stop,
            show_wdl: self.show_wdl,
            verbose_move_stats: self.verbose_move_stats,
            params: &self.params,
        };
        self.search.search(&mut self.net, game, &ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Play `turns` whole turns (piece move + duck) with the engine.
    fn play(engine: &mut PolicyEngine, turns: usize, limits: &SearchLimits) {
        let stop = AtomicBool::new(false);
        let mut game = Game::default();
        for _ in 0..turns {
            if game.result().is_over() {
                break;
            }
            let r = engine.search(&game, limits, &CommonOptions::default(), &stop);
            let uci = r.uci();
            assert!(
                game.push_uci(&uci),
                "illegal turn {uci} in {}",
                game.position().to_fen()
            );
        }
    }

    #[test]
    fn mcts_is_default_and_plays_legal_moves() {
        peaduck_core::init();
        let mut engine = PolicyEngine::new();
        assert_eq!(engine.mode, SearchMode::Mcts);
        let limits = SearchLimits {
            nodes: Some(50),
            ..Default::default()
        };
        play(&mut engine, 10, &limits);
    }

    #[test]
    fn searchless_mode_plays_legal_moves() {
        peaduck_core::init();
        let mut engine = PolicyEngine::new();
        assert!(engine.set_option("SearchMode", "searchless"));
        assert_eq!(engine.mode, SearchMode::Searchless);
        play(&mut engine, 40, &SearchLimits::default());
        assert!(engine.set_option("SearchMode", "Mcts"));
        assert_eq!(engine.mode, SearchMode::Mcts);
    }

    #[test]
    fn search_mode_option_lists_all_modes() {
        let line = SearchMode::uci_option(PolicyEngine::DEFAULT_MODE);
        assert_eq!(
            line,
            "option name SearchMode type combo default Mcts var Searchless var Mcts"
        );
    }

    #[test]
    fn move_probabilities_sum_to_one() {
        peaduck_core::init();
        let mut net = PolicyNet::embedded();
        let eval = net.evaluate(&Game::default());
        let probs = move_probabilities(&eval.moves);
        assert_eq!(probs.len(), 20);
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(probs.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn cp_mapping_is_monotone_and_bounded() {
        assert_eq!(q_to_cp(0.0), 0);
        assert!(q_to_cp(0.5) > 0 && q_to_cp(-0.5) < 0);
        assert!(q_to_cp(0.9) > q_to_cp(0.5));
        assert!(q_to_cp(1.0).abs() < 100_000);
    }

    #[test]
    fn search_params_are_uci_options() {
        let mut engine = PolicyEngine::new();
        let opts = engine.uci_options();
        let cpuct = crate::search::params::TUNABLES[0];
        assert_eq!(cpuct.0, "CPuct");
        assert!(opts.contains(&format!(
            "option name CPuct type spin default {} min {} max {}",
            cpuct.1, cpuct.2, cpuct.3
        )));
        assert!(opts
            .iter()
            .any(|o| o.starts_with("option name PolicyTempRoot type spin")));
        assert!(engine.set_option("CPuct", "300"));
        assert!(engine.set_option("fpureduction", "0"));
        assert_eq!(engine.params.cpuct, 3.0);
        assert_eq!(engine.params.fpu_reduction, 0.0);
        assert!(!engine.set_option("NoSuchOption", "1"));
    }
}
