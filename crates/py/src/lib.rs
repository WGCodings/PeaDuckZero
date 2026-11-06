//! `peaduck_rs` — Python bindings for the trainer.
//!
//! ```python
//! import peaduck_rs as pz
//! sp = pz.SearchSelfPlay(games=1024, simulations=64)
//! obs, legal = sp.requests()          # (N,23,8,8) float32, (N,256) int32
//! # policy (N,4736) → gather legal → (N,256); value (N,3)
//! sp.respond(legal_logits, value)     # one simulation for every game
//! chunk, games, positions, plies = sp.take_chunk()  # bytes in the data format
//!
//! rb = pz.Replay(capacity=1_000_000)
//! rb.add_chunk(chunk)
//! batch = rb.sample(512)              # dict of numpy arrays
//! ```
//!
//! Shapes are validated here and raise `ValueError` instead of panicking
//! (the release profile aborts on panic). The heavy calls release the GIL.

use numpy::{
    IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use peaduck_core::encoding::{ENCODING_VERSION, NUM_ACTIONS, OBS_PLANES, OBS_SIZE};
use peaduck_core::Game;
use peaduck_selfplay::record::{read_chunk, write_chunk, Termination};
use peaduck_selfplay::replay::{Replay as RustReplay, MAX_LEGAL};
use peaduck_selfplay::search::{GumbelConfig, Opening, PoolStats, SearchConfig, SearchPool};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn shaped<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    v.into_pyarray(py).reshape(shape.to_vec())
}

fn slice_of<'a>(
    a: &'a PyReadonlyArrayDyn<'_, f32>,
    expected: usize,
    what: &str,
) -> PyResult<&'a [f32]> {
    let s = a.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{what} must be a C-contiguous float32 array"))
    })?;
    if s.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {expected} values, got {} (shape {:?})",
            s.len(),
            a.shape()
        )));
    }
    Ok(s)
}

fn stats_dict<'py>(py: Python<'py>, s: &PoolStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("games", s.games_finished)?;
    d.set_item("plies", s.plies)?;
    d.set_item("full_plies", s.full_plies)?;
    d.set_item("white_wins", s.white_wins)?;
    d.set_item("black_wins", s.black_wins)?;
    d.set_item("draws", s.draws)?;
    let t = PyDict::new(py);
    for (i, &count) in s.terminations.iter().enumerate() {
        t.set_item(Termination::from_u8(i as u8).unwrap().name(), count)?;
    }
    d.set_item("terminations", t)?;
    Ok(d)
}

/// Self-play with Gumbel tree search on the real board. Every step of a
/// turn (piece move, duck move) is one search. Lock-step rounds:
/// `requests` → network → `respond`.
#[pyclass]
struct SearchSelfPlay {
    pool: SearchPool,
}

#[pymethods]
impl SearchSelfPlay {
    #[new]
    /// `max_plies`, `noise_plies`, `temp_decay_plies`: counted in steps (two
    /// per turn). `openings`: [(FEN or None for the start position, [UCI
    /// turns like "e2e4,e4d5"])]; a new game starts after 1..=all turns of a
    /// random line with probability `book_prob`, then plays
    /// 1..=`random_plies` random turns with probability `random_prob`.
    #[pyo3(signature = (games=1024, simulations=64, considered=16, gumbel_scale=1.0, c_visit=50.0, c_scale=0.1, max_plies=1024, seed=0, threads=0, noise_plies=0, late_gumbel_scale=0.0, fast_simulations=0, full_prob=0.25, fast_considered=8, openings=None, book_prob=0.0, random_prob=0.0, random_plies=0, temperature=0.0, temp_decay_plies=0, temp_endgame=0.0, temp_value_cutoff=0.1, root_explore=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        simulations: usize,
        considered: usize,
        gumbel_scale: f32,
        c_visit: f32,
        c_scale: f32,
        max_plies: usize,
        seed: u64,
        threads: usize,
        noise_plies: usize,
        late_gumbel_scale: f32,
        fast_simulations: usize,
        full_prob: f32,
        fast_considered: usize,
        openings: Option<Vec<(Option<String>, Vec<String>)>>,
        book_prob: f32,
        random_prob: f32,
        random_plies: usize,
        temperature: f32,
        temp_decay_plies: usize,
        temp_endgame: f32,
        temp_value_cutoff: f32,
        root_explore: f32,
    ) -> PyResult<Self> {
        if games == 0 || considered == 0 || simulations == 0 {
            return Err(PyValueError::new_err(
                "games, simulations and considered must be > 0",
            ));
        }
        peaduck_core::init();
        let openings = openings
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, (fen, turns))| {
                Opening::new(fen.as_deref(), turns)
                    .map_err(|e| PyValueError::new_err(format!("opening {i}: {e}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(SearchSelfPlay {
            pool: SearchPool::new(SearchConfig {
                games,
                simulations,
                max_plies,
                gumbel: GumbelConfig {
                    considered,
                    gumbel_scale,
                    c_visit,
                    c_scale,
                },
                seed,
                threads,
                noise_plies,
                late_gumbel_scale,
                fast_simulations,
                full_prob: full_prob.clamp(0.0, 1.0),
                fast_considered,
                openings: std::sync::Arc::new(openings),
                book_prob: book_prob.clamp(0.0, 1.0),
                random_prob: random_prob.clamp(0.0, 1.0),
                random_plies,
                temperature: temperature.max(0.0),
                temp_decay_plies,
                temp_endgame: temp_endgame.max(0.0),
                temp_value_cutoff: temp_value_cutoff.max(0.0),
                root_explore: root_explore.clamp(0.0, 1.0),
            }),
        })
    }

    #[getter]
    fn num_games(&self) -> usize {
        self.pool.num_games()
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    /// Positions to evaluate: (obs (N,23,8,8) float32, legal (N,256) int32
    /// policy indices of their legal steps, padded with 0).
    #[allow(clippy::type_complexity)]
    fn requests<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<i32>>)> {
        let n = self.pool.num_games();
        let pool = &self.pool;
        let (obs, legal) = py.detach(|| {
            let mut obs = vec![0.0f32; n * OBS_SIZE];
            let mut legal = vec![0i32; n * MAX_LEGAL];
            pool.requests(&mut obs, &mut legal);
            (obs, legal)
        });
        Ok((
            shaped(py, obs, &[n, OBS_PLANES, 8, 8])?,
            shaped(py, legal, &[n, MAX_LEGAL])?,
        ))
    }

    /// Like `requests`, but written into caller-owned C-contiguous arrays
    /// (page-locked buffers from the trainer).
    fn requests_into(
        &self,
        py: Python<'_>,
        mut obs: PyReadwriteArrayDyn<'_, f32>,
        mut legal: PyReadwriteArrayDyn<'_, i32>,
    ) -> PyResult<()> {
        let n = self.pool.num_games();
        let o = obs
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("obs must be a C-contiguous float32 array"))?;
        let l = legal
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("legal must be a C-contiguous int32 array"))?;
        if o.len() != n * OBS_SIZE || l.len() != n * MAX_LEGAL {
            return Err(PyValueError::new_err(format!(
                "requests_into: expected {} obs and {} legal values, got {} and {}",
                n * OBS_SIZE,
                n * MAX_LEGAL,
                o.len(),
                l.len()
            )));
        }
        let pool = &self.pool;
        py.detach(|| pool.requests(o, l));
        Ok(())
    }

    /// Legal-step logits (N,256) gathered at `legal`, value logits (N,3).
    /// Runs one simulation step for every game; returns games finished.
    fn respond(
        &mut self,
        py: Python<'_>,
        legal_logits: PyReadonlyArrayDyn<'_, f32>,
        value: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<usize> {
        let n = self.pool.num_games();
        let l = slice_of(&legal_logits, n * MAX_LEGAL, "legal_logits")?;
        let v = slice_of(&value, n * 3, "value")?;
        let pool = &mut self.pool;
        Ok(py.detach(|| pool.respond(l, v)))
    }

    /// Finished games as one data chunk: (bytes, games, positions, plies);
    /// positions = training samples (full-search steps), plies = all steps.
    fn take_chunk<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyBytes>, usize, usize, usize) {
        let games = self.pool.take_finished();
        let plies = games.iter().map(|g| g.plies.len()).sum();
        let positions = games
            .iter()
            .flat_map(|g| &g.plies)
            .filter(|p| p.full_search)
            .count();
        let bytes = py.detach(|| write_chunk(&games));
        (PyBytes::new(py, &bytes), games.len(), positions, plies)
    }

    fn plies_in_progress(&self) -> usize {
        self.pool.plies_in_progress()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_dict(py, self.pool.stats())
    }
}

#[pyclass]
struct Replay {
    inner: RustReplay,
}

#[pymethods]
impl Replay {
    #[new]
    #[pyo3(signature = (capacity=1_000_000, seed=0))]
    fn new(capacity: usize, seed: u64) -> Self {
        peaduck_core::init();
        Replay {
            inner: RustReplay::new(capacity, seed),
        }
    }

    /// Add a data chunk; returns the number of games added.
    fn add_chunk(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.detach(|| inner.add_chunk(data))
            .map_err(PyValueError::new_err)
    }

    /// Training samples held.
    #[getter]
    fn positions(&self) -> usize {
        self.inner.positions()
    }

    #[getter]
    fn games(&self) -> usize {
        self.inner.games()
    }

    #[getter]
    fn total_added(&self) -> u64 {
        self.inner.total_added()
    }

    /// Sample a training batch. Keys: obs (B,23,8,8), policy_index /
    /// policy_prob (B,256), value_target (B,) int64 (0 win, 1 draw, 2 loss),
    /// value_search (B,3) search W/D/L, value_search_mask (B,), duck_step (B,).
    fn sample<'py>(&mut self, py: Python<'py>, batch: usize) -> PyResult<Bound<'py, PyDict>> {
        if self.inner.positions() == 0 {
            return Err(PyValueError::new_err("replay buffer is empty"));
        }
        let inner = &mut self.inner;
        let b = py.detach(|| inner.sample(batch));
        let d = PyDict::new(py);
        d.set_item("obs", shaped(py, b.obs, &[batch, OBS_PLANES, 8, 8])?)?;
        d.set_item(
            "policy_index",
            shaped(py, b.policy_index, &[batch, MAX_LEGAL])?,
        )?;
        d.set_item(
            "policy_prob",
            shaped(py, b.policy_prob, &[batch, MAX_LEGAL])?,
        )?;
        d.set_item("value_target", shaped(py, b.value_target, &[batch])?)?;
        d.set_item("value_search", shaped(py, b.value_search, &[batch, 3])?)?;
        d.set_item(
            "value_search_mask",
            shaped(py, b.value_search_mask, &[batch])?,
        )?;
        d.set_item("duck_step", shaped(py, b.duck_step, &[batch])?)?;
        Ok(d)
    }
}

/// Encode a position given by FEN and UCI turns (`e2e4,e4d5`, or a single
/// step) exactly like the engine does: (23, 8, 8) float32 and the policy
/// indices of the legal steps.
#[pyfunction]
#[pyo3(signature = (fen=None, moves=Vec::new()))]
#[allow(clippy::type_complexity)]
fn encode_position<'py>(
    py: Python<'py>,
    fen: Option<&str>,
    moves: Vec<String>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Vec<usize>)> {
    peaduck_core::init();
    let mut g = match fen {
        Some(f) => Game::from_fen(f).map_err(|e| PyValueError::new_err(e.to_string()))?,
        None => Game::default(),
    };
    for m in &moves {
        if !g.push_uci(m) && !g.push_step(m) {
            return Err(PyValueError::new_err(format!("illegal move {m}")));
        }
    }
    let mut v = vec![0.0f32; OBS_SIZE];
    peaduck_core::encoding::encode_game(&g, &mut v);
    let legal = peaduck_core::encoding::legal_actions(g.position())
        .into_iter()
        .map(|(a, _)| a)
        .collect();
    Ok((shaped(py, v, &[OBS_PLANES, 8, 8])?, legal))
}

/// The position after `moves` (UCI turns, or a single step at the end) from
/// `fen` (None = start position), for GUIs: a dict with `fen`, `duck_step`,
/// `white_to_move`, `legal` (UCI strings of the legal next steps: piece
/// moves, or duck squares in the duck step), `result` ("*", "1-0", "0-1",
/// "1/2-1/2") and `reason`.
#[pyfunction]
#[pyo3(signature = (fen=None, moves=Vec::new()))]
fn position_info<'py>(
    py: Python<'py>,
    fen: Option<&str>,
    moves: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    peaduck_core::init();
    let mut g = match fen {
        Some(f) => Game::from_fen(f).map_err(|e| PyValueError::new_err(e.to_string()))?,
        None => Game::default(),
    };
    for m in &moves {
        if !g.push_uci(m) && !g.push_step(m) {
            return Err(PyValueError::new_err(format!("illegal move {m}")));
        }
    }
    let pos = g.position();
    let result = g.result();
    let d = PyDict::new(py);
    d.set_item("fen", pos.to_fen())?;
    d.set_item("duck_step", pos.is_duck_step())?;
    d.set_item("white_to_move", pos.side_to_move() == peaduck_core::Color::White)?;
    let legal: Vec<String> = if result.is_over() {
        Vec::new()
    } else {
        pos.legal_moves().iter().map(|m| m.to_uci()).collect()
    };
    d.set_item("legal", legal)?;
    d.set_item("result", result.pgn())?;
    let reason = match result {
        peaduck_core::GameResult::Ongoing => String::new(),
        peaduck_core::GameResult::Win(_, r) => format!("{r:?}"),
        peaduck_core::GameResult::Draw(r) => format!("{r:?}"),
    };
    d.set_item("reason", reason)?;
    Ok(d)
}

/// Zobrist keys of every position in a data chunk (unique-position counts).
#[pyfunction]
fn chunk_position_keys<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArrayDyn<u64>>> {
    peaduck_core::init();
    let games = read_chunk(data).map_err(PyValueError::new_err)?;
    let keys: Vec<u64> = games
        .iter()
        .flat_map(|g| g.plies.iter().map(|p| p.position().hash()))
        .collect();
    let n = keys.len();
    shaped(py, keys, &[n])
}

#[pymodule]
fn peaduck_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchSelfPlay>()?;
    m.add_class::<Replay>()?;
    m.add_function(wrap_pyfunction!(encode_position, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_position_keys, m)?)?;
    m.add_function(wrap_pyfunction!(position_info, m)?)?;
    m.add("ENCODING_VERSION", ENCODING_VERSION)?;
    m.add("MAX_LEGAL", MAX_LEGAL)?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("OBS_PLANES", OBS_PLANES)?;
    Ok(())
}
