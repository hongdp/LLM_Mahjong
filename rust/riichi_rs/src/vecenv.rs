//! Vectorized self-play driver: K interleaved single-deal games whose pending
//! decisions are exposed as one observation batch per round. Mirrors
//! parallel_rollout._worker_vectorized + selfplay.play_game_gen + _package_game
//! (mirror self-play: four learner seats, one temperature, native action space).
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::encoder::{encode_v1r, legal_mask, potential, ACTION_DIM, N_PLANES, N_SCALARS};
use crate::table::{resolve_claims, Table};

struct StepRec {
    planes: Vec<f32>,
    scalars: [f32; N_SCALARS],
    mask: Vec<bool>,
    action: i64,
    logprob: f32,
    reward: f64,
    phi: f64,
}

#[derive(Clone)]
struct RowRef {
    game: usize,
    seat: usize,
    lookup: Vec<(usize, usize)>,   // (slot, action index)
    actions: Vec<String>,
    interrupt: bool,
}

enum Phase {
    Turn,
    Interrupt,
}

struct Game {
    table: Table,
    seed: u64,
    traj: [Vec<StepRec>; 4],
    phase: Phase,
    guard: usize,
    // per-round decision bookkeeping
    rows: Vec<usize>,               // indices into VecEnv.rows for this game this round
    pending_steps: Vec<(usize, StepRec)>,   // (seat, recorded step) awaiting reward
}

struct Finished {
    seed: u64,
    episodes: Vec<(usize, Vec<StepRec>)>,   // (seat, steps)
    result: String,
    riichi: [bool; 4],
    n_melds: [usize; 4],
    n_discards: usize,
    points: [i64; 4],
    start_points: [i64; 4],
}

#[pyclass(name = "VecEnv")]
pub struct VecEnv {
    queue: std::collections::VecDeque<u64>,
    active: Vec<Option<Game>>,
    k: usize,
    gamma: f64,
    shaping: bool,
    shaping_scale: f64,
    randomize_round: bool,
    rows: Vec<RowRef>,
    finished: Vec<Finished>,
    n_started: usize,
}

impl VecEnv {
    fn start_game(&mut self, slot: usize, seed: u64) {
        let table = Table::new_seeded(seed, self.randomize_round);
        self.active[slot] = Some(Game {
            table,
            seed,
            traj: Default::default(),
            phase: Phase::Turn,
            guard: 0,
            rows: Vec::new(),
            pending_steps: Vec::new(),
        });
        self.n_started += 1;
    }

    fn fill_slots(&mut self) {
        for slot in 0..self.k {
            if self.active[slot].is_none() {
                if let Some(seed) = self.queue.pop_front() {
                    self.start_game(slot, seed);
                }
            }
        }
    }

    fn record(&self, table: &Table, seat: usize) -> (Vec<f32>, [f32; N_SCALARS], f64) {
        let mut planes = vec![0f32; N_PLANES * 34];
        let mut scalars = [0f32; N_SCALARS];
        encode_v1r(table, seat, &mut planes, &mut scalars);
        let phi = if self.shaping { potential(table, seat) } else { 0.0 };
        (planes, scalars, phi)
    }

    /// Bring a game to a state with pending decisions (or finish it).
    fn settle_to_decision(&mut self, slot: usize) {
        loop {
            let (finished, is_turn, guard, turn) = {
                let g = self.active[slot].as_ref().unwrap();
                (g.table.finished, matches!(g.phase, Phase::Turn), g.guard, g.table.turn)
            };
            if finished {
                self.finish(slot);
                return;
            }
            if is_turn {
                {
                    if guard >= 600 {
                        self.finish(slot);
                        return;
                    }
                    let pid = turn;
                    let actions = self.active[slot].as_ref().unwrap().table.get_legal_actions(pid);
                    if actions.is_empty() {
                        self.finish(slot);
                        return;
                    }
                    let (mask, lookup) = legal_mask(&actions);
                    let (planes, scalars, phi) = {
                        let g = self.active[slot].as_ref().unwrap();
                        self.record(&g.table, pid)
                    };
                    let g = self.active[slot].as_mut().unwrap();
                    g.guard += 1;
                    g.pending_steps.push((pid, StepRec { planes, scalars, mask: mask.to_vec(), action: 0, logprob: 0.0, reward: 0.0, phi }));
                    self.rows.push(RowRef { game: slot, seat: pid, lookup, actions, interrupt: false });
                    let idx = self.rows.len() - 1;
                    self.active[slot].as_mut().unwrap().rows.push(idx);
                    return;
                }
            } else {
                {
                    // build the window's requests: seats with more than one option
                    let pid = {
                        let g = self.active[slot].as_ref().unwrap();
                        g.table.last_discarder.or_else(|| g.table.pending_kan.as_ref().map(|k| k.player)).unwrap_or(g.table.turn)
                    };
                    let mut any = false;
                    for off in 1..4 {
                        let other = (pid + off) % 4;
                        let g = self.active[slot].as_ref().unwrap();
                        let opts = g.table.get_interrupt_actions(other);
                        if opts.len() <= 1 {
                            continue;
                        }
                        any = true;
                        let (mask, lookup) = legal_mask(&opts);
                        let (planes, scalars, phi) = self.record(&g.table, other);
                        let g = self.active[slot].as_mut().unwrap();
                        g.pending_steps.push((other, StepRec { planes, scalars, mask: mask.to_vec(), action: 0, logprob: 0.0, reward: 0.0, phi }));
                        self.rows.push(RowRef { game: slot, seat: other, lookup, actions: opts, interrupt: true });
                        let idx = self.rows.len() - 1;
                        self.active[slot].as_mut().unwrap().rows.push(idx);
                    }
                    if any {
                        return;
                    }
                    // nobody can act: resolve the empty window like the Python driver
                    let g = self.active[slot].as_mut().unwrap();
                    if g.table.pending_kan.is_some() {
                        g.table.resolve_pending_kan();
                    } else if g.table.advance_turn() {
                        self.finish(slot);
                        return;
                    }
                    let g = self.active[slot].as_mut().unwrap();
                    g.phase = Phase::Turn;
                }
            }
        }
    }

    fn finish(&mut self, slot: usize) {
        let g = self.active[slot].take().unwrap();
        let t = &g.table;
        let mut episodes = Vec::new();
        let mut traj = g.traj;
        if let Some(fr) = t.final_rewards {
            for p in 0..4 {
                if let Some(last) = traj[p].last_mut() {
                    last.reward += fr[p];
                }
            }
        }
        for p in 0..4 {
            let steps = std::mem::take(&mut traj[p]);
            if !steps.is_empty() {
                episodes.push((p, steps));
            }
        }
        self.finished.push(Finished {
            seed: g.seed,
            episodes,
            result: t.result_summary.clone(),
            riichi: t.riichi,
            n_melds: [t.melds[0].len(), t.melds[1].len(), t.melds[2].len(), t.melds[3].len()],
            n_discards: t.discard_count.iter().sum(),
            points: t.points,
            start_points: t.start_points,
        });
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (seeds, k, gamma=0.995, shaping=false, shaping_scale=1.0, randomize_round=true))]
    fn new(seeds: Vec<u64>, k: usize, gamma: f64, shaping: bool, shaping_scale: f64, randomize_round: bool) -> Self {
        let mut env = VecEnv {
            queue: seeds.into_iter().collect(),
            active: (0..k).map(|_| None).collect(),
            k,
            gamma,
            shaping,
            shaping_scale,
            randomize_round,
            rows: Vec::new(),
            finished: Vec::new(),
            n_started: 0,
        };
        env.fill_slots();
        for slot in 0..k {
            if env.active[slot].is_some() {
                env.settle_to_decision(slot);
            }
        }
        env
    }

    fn done(&self) -> bool {
        self.queue.is_empty() && self.active.iter().all(|g| g.is_none())
    }

    /// Pending decisions: (planes [B, 21*34] f32, scalars [B, 20] f32, mask [B, 374] bool, seats [B], games [B]).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
        let b = self.rows.len();
        let mut planes = Vec::with_capacity(b * N_PLANES * 34);
        let mut scalars = Vec::with_capacity(b * N_SCALARS);
        let mut mask = Vec::with_capacity(b * ACTION_DIM);
        let mut seats = Vec::with_capacity(b);
        let mut games = Vec::with_capacity(b);
        for r in &self.rows {
            let g = self.active[r.game].as_ref().unwrap();
            // the row's step record is the LAST pending step for that seat
            let rec = g.pending_steps.iter().rev().find(|(s, _)| *s == r.seat).map(|(_, rec)| rec).unwrap();
            planes.extend_from_slice(&rec.planes);
            scalars.extend_from_slice(&rec.scalars);
            mask.extend_from_slice(&rec.mask);
            seats.push(r.seat as i32);
            games.push(r.game as i32);
        }
        let planes = PyArray1::from_vec_bound(py, planes).reshape([b, N_PLANES * 34])?;
        let scalars = PyArray1::from_vec_bound(py, scalars).reshape([b, N_SCALARS])?;
        let mask = PyArray1::from_vec_bound(py, mask).reshape([b, ACTION_DIM])?;
        Ok((planes, scalars, mask, seats.into_pyarray_bound(py), games.into_pyarray_bound(py)))
    }

    /// Apply one action (slot index) + its logprob per pending row, advance every game.
    fn step(&mut self, actions: Vec<i64>, logprobs: Vec<f32>) -> PyResult<()> {
        if actions.len() != self.rows.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("actions length != pending rows"));
        }
        let rows = std::mem::take(&mut self.rows);
        // group rows by game, preserving order
        let mut by_game: Vec<Vec<usize>> = vec![Vec::new(); self.k];
        for (i, r) in rows.iter().enumerate() {
            by_game[r.game].push(i);
        }
        for slot in 0..self.k {
            if by_game[slot].is_empty() {
                continue;
            }
            let idxs = &by_game[slot];
            let g = self.active[slot].as_mut().unwrap();
            let first = &rows[idxs[0]];
            if !first.interrupt {
                // turn decision (exactly one row)
                let i = idxs[0];
                let slot_idx = actions[i];
                let xml = first.lookup.iter().find(|(s, _)| *s as i64 == slot_idx)
                    .map(|(_, a)| first.actions[*a].clone())
                    .unwrap_or_else(|| "<action type=\"skip\" />".to_string());
                let (seat, mut rec) = g.pending_steps.pop().unwrap();
                rec.action = slot_idx;
                rec.logprob = logprobs[i];
                let (rewards, done, info) = g.table.step(seat, &xml);
                rec.reward = rewards[seat];
                g.traj[seat].push(rec);
                g.rows.clear();
                if done {
                    self.finish(slot);
                    continue;
                }
                if info.discarded || info.chankan.is_some() {
                    g.phase = Phase::Interrupt;
                } else {
                    g.phase = Phase::Turn;
                }
            } else {
                // interrupt window: all requested seats answered this round
                let mut cands: Vec<(usize, String)> = Vec::with_capacity(idxs.len());
                let mut recs: Vec<(usize, StepRec)> = std::mem::take(&mut g.pending_steps);
                for &i in idxs {
                    let r = &rows[i];
                    let xml = r.lookup.iter().find(|(s, _)| *s as i64 == actions[i])
                        .map(|(_, a)| r.actions[*a].clone())
                        .unwrap_or_else(|| "<action type=\"skip\" />".to_string());
                    cands.push((r.seat, xml));
                }
                let (executed, done, rews) = resolve_claims(&mut g.table, &cands);
                for (j, &i) in idxs.iter().enumerate() {
                    let (seat, mut rec) = recs.remove(0);
                    debug_assert_eq!(seat, rows[i].seat);
                    rec.action = actions[i];
                    rec.logprob = logprobs[i];
                    rec.reward = rews[j];
                    g.traj[seat].push(rec);
                }
                g.rows.clear();
                if done {
                    self.finish(slot);
                    continue;
                }
                if executed.is_empty() {
                    if g.table.pending_kan.is_some() {
                        g.table.resolve_pending_kan();
                    } else if g.table.advance_turn() {
                        self.finish(slot);
                        continue;
                    }
                }
                g.phase = Phase::Turn;
            }
        }
        self.fill_slots();
        for slot in 0..self.k {
            if self.active[slot].is_some() {
                self.settle_to_decision(slot);
            }
        }
        Ok(())
    }

    /// Finished games as _package_game-style dicts (planes shipped as float32; the caller
    /// downcasts to float16 like the Python packer).
    fn drain_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty_bound(py);
        let finished = std::mem::take(&mut self.finished);
        for f in finished {
            let d = PyDict::new_bound(py);
            let eps = PyList::empty_bound(py);
            for (seat, mut steps) in f.episodes {
                let n = steps.len();
                if self.shaping {
                    for i in 0..n {
                        let nxt = if i + 1 < n { steps[i + 1].phi } else { 0.0 };
                        let cur = steps[i].phi;
                        steps[i].reward += self.shaping_scale * (self.gamma * nxt - cur);
                    }
                }
                let mut rets = vec![0f32; n];
                let mut r = 0.0f64;
                for i in (0..n).rev() {
                    r = steps[i].reward + self.gamma * r;
                    rets[i] = r as f32;
                }
                let mut planes = Vec::with_capacity(n * N_PLANES * 34);
                let mut scal = Vec::with_capacity(n * N_SCALARS);
                let mut mask = Vec::with_capacity(n * ACTION_DIM);
                let mut acts = Vec::with_capacity(n);
                let mut lps = Vec::with_capacity(n);
                let mut rews = Vec::with_capacity(n);
                for s in &steps {
                    planes.extend_from_slice(&s.planes);
                    scal.extend_from_slice(&s.scalars);
                    mask.extend_from_slice(&s.mask);
                    acts.push(s.action);
                    lps.push(s.logprob);
                    rews.push(s.reward as f32);
                }
                let e = PyDict::new_bound(py);
                e.set_item("planes", PyArray1::from_vec_bound(py, planes).reshape([n, N_PLANES, 34])?)?;
                e.set_item("scalars", PyArray1::from_vec_bound(py, scal).reshape([n, N_SCALARS])?)?;
                e.set_item("mask", PyArray1::from_vec_bound(py, mask).reshape([n, ACTION_DIM])?)?;
                e.set_item("actions", acts.into_pyarray_bound(py))?;
                e.set_item("old_logprobs", lps.into_pyarray_bound(py))?;
                e.set_item("returns", rets.into_pyarray_bound(py))?;
                e.set_item("rewards", rews.into_pyarray_bound(py))?;
                e.set_item("key", (f.seed, seat))?;
                eps.append(e)?;
            }
            d.set_item("episodes", eps)?;
            d.set_item("result", f.result)?;
            d.set_item("riichi", f.riichi.to_vec())?;
            d.set_item("n_melds", f.n_melds.to_vec())?;
            d.set_item("n_discards", f.n_discards)?;
            d.set_item("points", f.points.to_vec())?;
            d.set_item("start_points", f.start_points.to_vec())?;
            d.set_item("learner_seats", vec![0usize, 1, 2, 3])?;
            d.set_item("league", PyDict::new_bound(py))?;
            d.set_item("seed", f.seed)?;
            out.append(d)?;
        }
        Ok(out)
    }

    #[getter]
    fn n_pending(&self) -> usize {
        self.rows.len()
    }
}
