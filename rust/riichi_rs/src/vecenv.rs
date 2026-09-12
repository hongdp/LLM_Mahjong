//! Vectorized self-play driver: K interleaved single-deal games whose pending
//! decisions are exposed as one observation batch per round. Mirrors
//! parallel_rollout._worker_vectorized + selfplay.play_game_gen + _package_game
//! (mirror self-play: four learner seats, one temperature, native action space).
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rayon::prelude::*;

use crate::encoder::{encode, legal_mask, planes_of, potential, quantize_planes, scalars_of, ACTION_DIM};
use crate::hanchan::{DealFact, MatchResult, MatchState};
use crate::table::{resolve_claims, Table, REWARD_SCALE};

#[derive(Clone)]
struct Cfg {
    shaping: bool,
    variant: String,
    n_planes: usize,
    n_scalars: usize,
}

struct StepRec {
    planes: Vec<u8>,   // quantised (see encoder::PLANE_Q)
    scalars: Vec<f32>,
    mask: Vec<bool>,
    action: i64,
    logprob: f32,
    reward: f64,
    phi: f64,
}

#[derive(Clone)]
struct RowRef {
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
    // per-round decision bookkeeping (this game's pending rows, in request order)
    rows: Vec<RowRef>,
    pending_steps: Vec<(usize, StepRec)>,   // (seat, recorded step) awaiting reward
    // hanchan mode (play_hanchan_gen): match bookkeeping, per-deal facts, and
    // where each seat's trajectory stood when the current deal began
    ms: Option<MatchState>,
    deal_facts: Vec<DealFact>,
    deal_start_len: [usize; 4],
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
    hanchan: Option<MatchResult>,
    deals: Vec<DealFact>,
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
    variant: String,
    finished: Vec<Finished>,
    n_started: usize,
    hanchan: bool,
    max_deals: usize,
    houjuu_extra: f64,
}

fn record(table: &Table, seat: usize, cfg: &Cfg) -> (Vec<u8>, Vec<f32>, f64) {
    let mut planes_f = vec![0f32; cfg.n_planes * 34];
    let mut scalars = vec![0f32; cfg.n_scalars];
    encode(table, seat, &cfg.variant, &mut planes_f, &mut scalars);
    let mut planes = vec![0u8; cfg.n_planes * 34];
    quantize_planes(&planes_f, &mut planes);
    let phi = if cfg.shaping { potential(table, seat) } else { 0.0 };
    (planes, scalars, phi)
}

fn deal_fact(t: &Table) -> DealFact {
    DealFact {
        result: t.result_summary.clone(),
        riichi: t.riichi,
        n_melds: [t.melds[0].len(), t.melds[1].len(), t.melds[2].len(), t.melds[3].len()],
        n_discards: t.discard_count.iter().sum(),
        points: t.points,
        start_points: t.start_points,
    }
}

/// A deal ended. Single deal: the game is over. Hanchan: pay this deal's driver
/// point delta on each seat's last step of the deal (play_hanchan_gen, credit=None),
/// advance the match and either start the next deal (returns false) or end (true).
fn deal_done(g: &mut Game) -> bool {
    let Some(ms) = g.ms.as_mut() else { return true };
    let before = ms.points;
    ms.settle(&g.table);
    for p in 0..4 {
        if g.traj[p].len() > g.deal_start_len[p] {
            if let Some(last) = g.traj[p].last_mut() {
                last.reward += (ms.points[p] - before[p]) as f64 * REWARD_SCALE;
            }
        }
    }
    g.deal_facts.push(deal_fact(&g.table));
    if ms.done {
        return true;
    }
    let (dealer, rw, pts, kyo) = ms.begin_deal();
    let honba = ms.honba;
    let n = ms.n as u64;
    let hx = g.table.houjuu_extra;
    g.table = Table::new_hanchan(g.seed.wrapping_mul(1_000_003).wrapping_add(n), dealer, rw, pts, kyo, honba);
    g.table.houjuu_extra = hx;
    g.phase = Phase::Turn;
    g.guard = 0;
    g.rows.clear();
    g.pending_steps.clear();
    for p in 0..4 {
        g.deal_start_len[p] = g.traj[p].len();
    }
    false
}

fn finish(g: Game) -> Finished {
    let t = &g.table;
    let mut traj = g.traj;
    let mut hanchan = None;
    let mut points = t.points;
    let mut start_points = t.start_points;
    if let Some(ms) = &g.ms {
        // terminal placement signal: UMA + leftover sticks only (the point margin
        // already arrived deal by deal) => per-seat return == final - 25000 + UMA
        let res = ms.result();
        for p in 0..4 {
            if let Some(last) = traj[p].last_mut() {
                last.reward += (res.uma_points[p] - (ms.points[p] - 25000)) as f64 * REWARD_SCALE;
            }
        }
        points = res.final_points;
        start_points = [25000; 4];
        hanchan = Some(res);
    } else if let Some(fr) = t.final_rewards {
        for p in 0..4 {
            if let Some(last) = traj[p].last_mut() {
                last.reward += fr[p];
            }
        }
    }
    let mut episodes = Vec::new();
    for p in 0..4 {
        let steps = std::mem::take(&mut traj[p]);
        if !steps.is_empty() {
            episodes.push((p, steps));
        }
    }
    Finished {
        seed: g.seed,
        episodes,
        result: t.result_summary.clone(),
        riichi: t.riichi,
        n_melds: [t.melds[0].len(), t.melds[1].len(), t.melds[2].len(), t.melds[3].len()],
        n_discards: t.discard_count.iter().sum(),
        points,
        start_points,
        hanchan,
        deals: g.deal_facts,
    }
}

/// Bring a game to a state with pending decisions; returns true when it finished instead.
fn settle_to_decision(g: &mut Game, cfg: &Cfg) -> bool {
    loop {
        if g.table.finished {
            return true;
        }
        match g.phase {
            Phase::Turn => {
                if g.guard >= 600 {
                    return true;
                }
                let pid = g.table.turn;
                let actions = g.table.get_legal_actions(pid);
                if actions.is_empty() {
                    return true;
                }
                let (mask, lookup) = legal_mask(&actions);
                let (planes, scalars, phi) = record(&g.table, pid, cfg);
                g.guard += 1;
                g.pending_steps.push((pid, StepRec { planes, scalars, mask: mask.to_vec(), action: 0, logprob: 0.0, reward: 0.0, phi }));
                g.rows.push(RowRef { seat: pid, lookup, actions, interrupt: false });
                return false;
            }
            Phase::Interrupt => {
                let pid = g.table.last_discarder.or_else(|| g.table.pending_kan.as_ref().map(|k| k.player)).unwrap_or(g.table.turn);
                let mut any = false;
                for off in 1..4 {
                    let other = (pid + off) % 4;
                    let opts = g.table.get_interrupt_actions(other);
                    if opts.len() <= 1 {
                        continue;
                    }
                    any = true;
                    let (mask, lookup) = legal_mask(&opts);
                    let (planes, scalars, phi) = record(&g.table, other, cfg);
                    g.pending_steps.push((other, StepRec { planes, scalars, mask: mask.to_vec(), action: 0, logprob: 0.0, reward: 0.0, phi }));
                    g.rows.push(RowRef { seat: other, lookup, actions: opts, interrupt: true });
                }
                if any {
                    return false;
                }
                if g.table.pending_kan.is_some() {
                    g.table.resolve_pending_kan();
                } else if g.table.advance_turn() {
                    return true;
                }
                g.phase = Phase::Turn;
            }
        }
    }
}

/// Apply this round's answers for one game; returns true when the game finished.
fn apply_round(g: &mut Game, actions: &[i64], logprobs: &[f32]) -> bool {
    let rows = std::mem::take(&mut g.rows);
    if rows.is_empty() {
        return false;
    }
    if !rows[0].interrupt {
        let r = &rows[0];
        let slot_idx = actions[0];
        let xml = r.lookup.iter().find(|(s, _)| *s as i64 == slot_idx)
            .map(|(_, a)| r.actions[*a].clone())
            .unwrap_or_else(|| "<action type=\"skip\" />".to_string());
        let (seat, mut rec) = g.pending_steps.pop().unwrap();
        rec.action = slot_idx;
        rec.logprob = logprobs[0];
        let (rewards, done, info) = g.table.step(seat, &xml);
        rec.reward = rewards[seat];
        g.traj[seat].push(rec);
        if done {
            return true;
        }
        g.phase = if info.discarded || info.chankan.is_some() { Phase::Interrupt } else { Phase::Turn };
        false
    } else {
        let mut cands: Vec<(usize, String)> = Vec::with_capacity(rows.len());
        let mut recs: Vec<(usize, StepRec)> = std::mem::take(&mut g.pending_steps);
        for (i, r) in rows.iter().enumerate() {
            let xml = r.lookup.iter().find(|(s, _)| *s as i64 == actions[i])
                .map(|(_, a)| r.actions[*a].clone())
                .unwrap_or_else(|| "<action type=\"skip\" />".to_string());
            cands.push((r.seat, xml));
        }
        let (executed, done, rews) = resolve_claims(&mut g.table, &cands);
        for (i, _r) in rows.iter().enumerate() {
            let (seat, mut rec) = recs.remove(0);
            rec.action = actions[i];
            rec.logprob = logprobs[i];
            rec.reward = rews[i];
            g.traj[seat].push(rec);
        }
        if done {
            return true;
        }
        if executed.is_empty() {
            if g.table.pending_kan.is_some() {
                g.table.resolve_pending_kan();
            } else if g.table.advance_turn() {
                return true;
            }
        }
        g.phase = Phase::Turn;
        false
    }
}

impl VecEnv {
    fn cfg(&self) -> Cfg {
        Cfg { shaping: self.shaping, variant: self.variant.clone(), n_planes: planes_of(&self.variant), n_scalars: scalars_of(&self.variant) }
    }

    fn fill_slots(&mut self) {
        for slot in 0..self.k {
            if self.active[slot].is_none() {
                if let Some(seed) = self.queue.pop_front() {
                    let (mut table, ms) = if self.hanchan {
                        let mut ms = MatchState::new(self.max_deals);
                        let (dealer, rw, pts, kyo) = ms.begin_deal();
                        let honba = ms.honba;
                        // play_hanchan_gen: random.seed(match_seed * 1000003 + n) per deal
                        (Table::new_hanchan(seed.wrapping_mul(1_000_003).wrapping_add(ms.n as u64), dealer, rw, pts, kyo, honba), Some(ms))
                    } else {
                        (Table::new_seeded(seed, self.randomize_round), None)
                    };
                    table.houjuu_extra = self.houjuu_extra;
                    self.active[slot] = Some(Game {
                        table, seed, traj: Default::default(), phase: Phase::Turn, guard: 0,
                        rows: Vec::new(), pending_steps: Vec::new(),
                        ms, deal_facts: Vec::new(), deal_start_len: [0; 4],
                    });
                    self.n_started += 1;
                }
            }
        }
    }

    /// Settle every active game without pending rows (parallel), collecting finished ones.
    fn settle_all(&mut self) {
        let cfg = self.cfg();
        let finished: Vec<Finished> = self.active.par_iter_mut()
            .filter_map(|slot| {
                let g = slot.as_mut()?;
                if !g.rows.is_empty() {
                    return None;
                }
                let mut done = settle_to_decision(g, &cfg);
                while done {
                    if deal_done(g) {
                        let g = slot.take().unwrap();
                        return Some(finish(g));
                    }
                    done = settle_to_decision(g, &cfg);
                }
                None
            })
            .collect();
        self.finished.extend(finished);
    }

    fn total_rows(&self) -> usize {
        self.active.iter().map(|g| g.as_ref().map_or(0, |g| g.rows.len())).sum()
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (seeds, k, gamma=0.995, shaping=false, shaping_scale=1.0, randomize_round=true, variant="v1r".to_string(), hanchan=false, max_deals=24, houjuu_extra=0.0))]
    fn new(seeds: Vec<u64>, k: usize, gamma: f64, shaping: bool, shaping_scale: f64, randomize_round: bool, variant: String, hanchan: bool, max_deals: usize, houjuu_extra: f64) -> PyResult<Self> {
        if hanchan && shaping {
            return Err(pyo3::exceptions::PyValueError::new_err("hanchan VecEnv: PBRS shaping is per-deal and not supported across a match"));
        }
        let mut env = VecEnv {
            queue: seeds.into_iter().collect(),
            active: (0..k).map(|_| None).collect(),
            k, gamma, shaping, shaping_scale, randomize_round, variant,
            finished: Vec::new(),
            n_started: 0,
            hanchan, max_deals, houjuu_extra,
        };
        env.fill_slots();
        env.settle_all();
        // games that finished instantly leave empty slots: keep filling
        while env.active.iter().any(|g| g.is_none()) && !env.queue.is_empty() {
            env.fill_slots();
            env.settle_all();
        }
        Ok(env)
    }

    /// PIMC rollouts: one game per supplied table (all active at once), each forced to
    /// take `first_actions[i]` for `seats[i]` first (a turn-phase action: discard /
    /// riichi / tsumo / kan), then played out by the caller's policy to the end of the
    /// deal. `seed` per game = its index, so drain_finished() keys match the input order.
    #[staticmethod]
    #[pyo3(signature = (tables, seats, first_actions, gamma=0.995, variant="v1r".to_string()))]
    fn from_tables(tables: Vec<PyRef<crate::pytable::PyTable>>, seats: Vec<usize>, first_actions: Vec<String>, gamma: f64, variant: String) -> PyResult<Self> {
        if tables.len() != seats.len() || tables.len() != first_actions.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("tables / seats / first_actions length mismatch"));
        }
        let k = tables.len().max(1);
        let mut env = VecEnv {
            queue: std::collections::VecDeque::new(),
            active: (0..k).map(|_| None).collect(),
            k, gamma, shaping: false, shaping_scale: 1.0, randomize_round: true, variant,
            finished: Vec::new(),
            n_started: 0,
            hanchan: false, max_deals: 24, houjuu_extra: 0.0,
        };
        for (i, (pt, (seat, xml))) in tables.iter().zip(seats.iter().zip(first_actions.iter())).enumerate() {
            let mut table = pt.table_clone();
            if table.finished {
                continue;
            }
            let (_rewards, done, info) = table.step(*seat, xml);
            let phase = if info.discarded || info.chankan.is_some() { Phase::Interrupt } else { Phase::Turn };
            let mut g = Game {
                table, seed: i as u64, traj: Default::default(), phase, guard: 0,
                rows: Vec::new(), pending_steps: Vec::new(),
                ms: None, deal_facts: Vec::new(), deal_start_len: [0; 4],
            };
            if done {
                env.finished.push(finish(g));
                continue;
            }
            // make sure the forced step leaves a consistent turn pointer for the loop
            g.guard = 0;
            env.active[i] = Some(g);
            env.n_started += 1;
        }
        env.settle_all();
        Ok(env)
    }

    fn done(&self) -> bool {
        self.queue.is_empty() && self.active.iter().all(|g| g.is_none())
    }

    /// Pending decisions: (planes [B, np*34] u8 quantised by PLANE_Q, scalars [B, ns] f32, mask [B, 374] bool, seats [B], games [B]).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
        let b = self.total_rows();
        let np = planes_of(&self.variant);
        let ns = scalars_of(&self.variant);
        let mut planes = Vec::with_capacity(b * np * 34);
        let mut scalars = Vec::with_capacity(b * ns);
        let mut mask = Vec::with_capacity(b * ACTION_DIM);
        let mut seats = Vec::with_capacity(b);
        let mut games = Vec::with_capacity(b);
        for (slot, g) in self.active.iter().enumerate() {
            let Some(g) = g else { continue };
            // pending_steps are in the same order as rows (turn: 1; interrupt: request order)
            let base = g.pending_steps.len() - g.rows.len();
            for (i, r) in g.rows.iter().enumerate() {
                let rec = &g.pending_steps[base + i].1;
                planes.extend_from_slice(&rec.planes);
                scalars.extend_from_slice(&rec.scalars);
                mask.extend_from_slice(&rec.mask);
                seats.push(r.seat as i32);
                games.push(slot as i32);
            }
        }
        let planes = PyArray1::from_vec_bound(py, planes).reshape([b, np * 34])?;
        let scalars = PyArray1::from_vec_bound(py, scalars).reshape([b, ns])?;
        let mask = PyArray1::from_vec_bound(py, mask).reshape([b, ACTION_DIM])?;
        Ok((planes, scalars, mask, seats.into_pyarray_bound(py), games.into_pyarray_bound(py)))
    }

    /// Apply one action (slot index) + its logprob per pending row (row order = observe()), advance every game.
    fn step(&mut self, py: Python<'_>, actions: Vec<i64>, logprobs: Vec<f32>) -> PyResult<()> {
        if actions.len() != self.total_rows() {
            return Err(pyo3::exceptions::PyValueError::new_err("actions length != pending rows"));
        }
        // per-slot offsets into the flat action arrays
        let mut offsets = Vec::with_capacity(self.k);
        let mut off = 0usize;
        for g in &self.active {
            let n = g.as_ref().map_or(0, |g| g.rows.len());
            offsets.push((off, n));
            off += n;
        }
        let cfg = self.cfg();
        let finished: Vec<Finished> = py.allow_threads(|| {
            self.active.par_iter_mut().zip(offsets.par_iter())
                .filter_map(|(slot, &(o, n))| {
                    let g = slot.as_mut()?;
                    if n == 0 {
                        return None;
                    }
                    let mut done = apply_round(g, &actions[o..o + n], &logprobs[o..o + n])
                        || settle_to_decision(g, &cfg);
                    while done {
                        if deal_done(g) {
                            let g = slot.take().unwrap();
                            return Some(finish(g));
                        }
                        done = settle_to_decision(g, &cfg);
                    }
                    None
                })
                .collect()
        });
        self.finished.extend(finished);
        self.fill_slots();
        self.settle_all();
        while self.active.iter().any(|g| g.is_none()) && !self.queue.is_empty() {
            self.fill_slots();
            self.settle_all();
        }
        Ok(())
    }

    /// Finished games as _package_game-style dicts (planes shipped as u8 quantised by
    /// PLANE_Q; the trainer widens them on the device — no float16 hop).
    fn drain_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty_bound(py);
        let np = planes_of(&self.variant);
        let ns = scalars_of(&self.variant);
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
                let mut planes = Vec::with_capacity(n * np * 34);
                let mut scal = Vec::with_capacity(n * ns);
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
                e.set_item("planes", PyArray1::from_vec_bound(py, planes).reshape([n, np, 34])?)?;
                e.set_item("scalars", PyArray1::from_vec_bound(py, scal).reshape([n, ns])?)?;
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
            if let Some(h) = &f.hanchan {
                let hd = PyDict::new_bound(py);
                hd.set_item("placements", h.placements.to_vec())?;
                hd.set_item("uma_points", h.uma_points.to_vec())?;
                hd.set_item("busted", h.busted)?;
                hd.set_item("n_deals", h.n_deals)?;
                d.set_item("hanchan", hd)?;
                let roles = PyDict::new_bound(py);
                roles.set_item("pure", true)?;
                roles.set_item("learner", 0usize)?;
                d.set_item("roles", roles)?;
                let deals = PyList::empty_bound(py);
                for df in &f.deals {
                    let dd = PyDict::new_bound(py);
                    dd.set_item("result", &df.result)?;
                    dd.set_item("riichi", df.riichi.to_vec())?;
                    dd.set_item("n_melds", df.n_melds.to_vec())?;
                    dd.set_item("n_discards", df.n_discards)?;
                    dd.set_item("points", df.points.to_vec())?;
                    dd.set_item("start_points", df.start_points.to_vec())?;
                    deals.append(dd)?;
                }
                d.set_item("deals", deals)?;
            } else {
                d.set_item("hanchan", py.None())?;
            }
            out.append(d)?;
        }
        Ok(out)
    }

    #[getter]
    fn n_pending(&self) -> usize {
        self.total_rows()
    }

    /// Deal seed per slot (-1 for an empty slot): lets the Python driver map observe()'s
    /// `games` column back to the deal seed (league seat plans are keyed by seed).
    fn slot_seeds(&self) -> Vec<i64> {
        self.active.iter().map(|g| g.as_ref().map_or(-1, |g| g.seed as i64)).collect()
    }
}
