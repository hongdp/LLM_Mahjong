//! Python bindings of `Table` (attribute access mirrors PyMahjongTable so the
//! parity tests and the Python encoder can read either engine).
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::table::{resolve_claims, MeldKind, StepInfo, Table};
use crate::tiles::*;

#[pyclass(name = "Table")]
pub struct PyTable {
    pub t: Table,
}

fn strs(v: &[Tile]) -> Vec<String> {
    v.iter().map(|&t| to_string(t)).collect()
}

fn info_dict(py: Python<'_>, info: &StepInfo) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("discarded", info.discarded)?;
    if let Some(t) = info.chankan {
        d.set_item("chankan", to_string(t))?;
    }
    d.set_item("interrupt", info.interrupt)?;
    d.set_item("winners", info.winners.clone())?;
    if let Some(a) = &info.abort {
        d.set_item("abort", a)?;
    }
    Ok(d.into())
}

#[pymethods]
impl PyTable {
    #[new]
    #[pyo3(signature = (seed, randomize_round=true))]
    fn new(seed: u64, randomize_round: bool) -> Self {
        PyTable { t: Table::new_seeded(seed, randomize_round) }
    }
    // ---- state ----
    #[getter] fn dealer(&self) -> usize { self.t.dealer }
    #[getter] fn round_wind_idx(&self) -> usize { self.t.round_wind_idx }
    #[getter] fn round_number(&self) -> usize { self.t.round_number }
    #[getter] fn turn(&self) -> usize { self.t.turn }
    #[getter] fn points(&self) -> Vec<i64> { self.t.points.to_vec() }
    #[getter] fn start_points(&self) -> Vec<i64> { self.t.start_points.to_vec() }
    #[getter] fn kyotaku(&self) -> i64 { self.t.kyotaku }
    #[getter] fn honba(&self) -> i64 { self.t.honba }
    #[getter] fn wall(&self) -> Vec<String> { strs(&self.t.wall) }
    #[getter] fn wall_len(&self) -> usize { self.t.wall.len() }
    #[getter] fn dead_wall(&self) -> Vec<String> { strs(&self.t.dead_wall) }
    #[getter] fn dora_indicators(&self) -> Vec<String> { strs(&self.t.dora_indicators) }
    #[getter] fn ura_indicators(&self) -> Vec<String> { strs(&self.t.ura_indicators) }
    #[getter] fn hands(&self) -> Vec<Vec<String>> { (0..4).map(|p| strs(&self.t.hands[p])).collect() }
    #[getter] fn red(&self) -> Vec<Vec<u8>> { (0..4).map(|p| self.t.red[p].to_vec()).collect() }
    #[getter] fn last_drawn(&self) -> Vec<Option<String>> { (0..4).map(|p| self.t.last_drawn[p].map(to_string)).collect() }
    #[getter] fn last_drawn_red(&self) -> Vec<bool> { self.t.last_drawn_red.to_vec() }
    #[getter] fn last_discard(&self) -> Option<String> { self.t.last_discard.map(to_string) }
    #[getter] fn last_discard_red(&self) -> bool { self.t.last_discard_red }
    #[getter] fn last_discarder(&self) -> Option<usize> { self.t.last_discarder }
    #[getter] fn finished(&self) -> bool { self.t.finished }
    #[getter] fn result_summary(&self) -> String { self.t.result_summary.clone() }
    #[getter] fn final_rewards(&self) -> Option<Vec<f64>> { self.t.final_rewards.map(|f| f.to_vec()) }
    #[getter] fn riichi(&self) -> Vec<bool> { self.t.riichi.to_vec() }
    #[getter] fn ippatsu(&self) -> Vec<bool> { self.t.ippatsu.to_vec() }
    #[getter] fn discard_count(&self) -> Vec<usize> { self.t.discard_count.to_vec() }
    #[getter] fn kan_count(&self) -> usize { self.t.kan_count }
    #[getter] fn any_call(&self) -> bool { self.t.any_call }
    #[getter] fn temp_furiten(&self) -> Vec<bool> { self.t.temp_furiten.to_vec() }
    #[getter] fn perm_furiten(&self) -> Vec<bool> { self.t.perm_furiten.to_vec() }
    #[getter] fn pending_kan(&self) -> Option<(usize, String, bool, bool)> {
        self.t.pending_kan.as_ref().map(|k| (k.player, to_string(k.tile), k.ankan, k.red))
    }
    /// Visible rivers as the Python engine spells them ('0m', trailing '*' for the riichi tile).
    #[getter]
    fn discards(&self) -> Vec<Vec<String>> {
        (0..4).map(|p| self.t.discards[p].iter().map(|(t, m)| format!("{}{}", to_string(*t), if *m { "*" } else { "" })).collect()).collect()
    }
    #[getter] fn furiten_river(&self) -> Vec<Vec<String>> { (0..4).map(|p| strs(&self.t.furiten_river[p])).collect() }
    #[getter]
    fn river_events(&self) -> Vec<Vec<(String, bool, bool, bool, usize)>> {
        (0..4).map(|p| self.t.river_events[p].iter().map(|e| (to_string(e.tile), e.tsumogiri, e.riichi, e.called, e.idx)).collect()).collect()
    }
    #[getter]
    fn melds(&self, py: Python<'_>) -> PyResult<Vec<Vec<PyObject>>> {
        let mut out = Vec::new();
        for p in 0..4 {
            let mut v = Vec::new();
            for m in &self.t.melds[p] {
                let d = PyDict::new_bound(py);
                let ty = match m.kind { MeldKind::Chi => "chi", MeldKind::Pon => "pon", MeldKind::Kan => "kan",
                                        MeldKind::Ankan => "ankan", MeldKind::Shouminkan => "shouminkan" };
                d.set_item("type", ty)?;
                d.set_item("tiles", strs(&m.tiles))?;
                d.set_item("opened", m.opened)?;
                d.set_item("red", m.red)?;
                if let Some(f) = m.from { d.set_item("from", f)?; }
                v.push(d.into());
            }
            out.push(v);
        }
        Ok(out)
    }
    // ---- game API ----
    fn get_legal_actions(&self, pid: usize) -> Vec<String> { self.t.get_legal_actions(pid) }
    fn get_interrupt_actions(&self, pid: usize) -> Vec<String> { self.t.get_interrupt_actions(pid) }
    fn step(&mut self, py: Python<'_>, pid: usize, action_xml: &str) -> PyResult<(Vec<f64>, bool, PyObject)> {
        let (r, done, info) = self.t.step(pid, action_xml);
        Ok((r.to_vec(), done, info_dict(py, &info)?))
    }
    fn step_ron(&mut self, py: Python<'_>, player_ids: Vec<usize>) -> PyResult<(Vec<f64>, bool, PyObject)> {
        let (r, done, info) = self.t.step_ron(&player_ids);
        Ok((r.to_vec(), done, info_dict(py, &info)?))
    }
    fn step_interrupt(&mut self, py: Python<'_>, pid: usize, action_xml: &str) -> PyResult<(Vec<f64>, bool, PyObject)> {
        let (r, done, info) = self.t.step_interrupt(pid, action_xml);
        Ok((r.to_vec(), done, info_dict(py, &info)?))
    }
    fn advance_turn(&mut self) -> bool { self.t.advance_turn() }
    fn resolve_pending_kan(&mut self) { self.t.resolve_pending_kan() }
    /// claims._resolve_claims: cands = [(seat, xml)] -> (executed seats, done, reward per cand)
    fn resolve_claims(&mut self, cands: Vec<(usize, String)>) -> (Vec<usize>, bool, Vec<f64>) {
        resolve_claims(&mut self.t, &cands)
    }
    fn waits(&self, pid: usize) -> Vec<String> { strs(&self.t.waits(pid)) }
    fn win_result(&self, py: Python<'_>, pid: usize, tile: &str, is_tsumo: bool, chankan: bool) -> PyResult<Option<PyObject>> {
        let t = parse(tile).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad tile"))?;
        match self.t.win_result(pid, norm(t), is_tsumo, chankan) {
            None => Ok(None),
            Some(r) => {
                let d = PyDict::new_bound(py);
                d.set_item("han", r.han)?; d.set_item("fu", r.fu)?; d.set_item("main", r.main)?;
                d.set_item("additional", r.additional)?;
                d.set_item("yaku", r.yaku.iter().map(|y| y.name()).collect::<Vec<_>>())?;
                Ok(Some(d.into()))
            }
        }
    }
}
