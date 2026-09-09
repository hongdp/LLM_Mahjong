//! Python bindings of `Table` (attribute access mirrors PyMahjongTable so the
//! parity tests and, later, the Python encoder can read either engine).
use pyo3::prelude::*;

use crate::table::Table;
use crate::tiles::*;

#[pyclass(name = "Table")]
pub struct PyTable {
    pub t: Table,
}

fn strs(v: &[Tile]) -> Vec<String> {
    v.iter().map(|&t| to_string(t)).collect()
}

#[pymethods]
impl PyTable {
    #[new]
    #[pyo3(signature = (seed, randomize_round=true))]
    fn new(seed: u64, randomize_round: bool) -> Self {
        PyTable { t: Table::new_seeded(seed, randomize_round) }
    }
    #[getter]
    fn dealer(&self) -> usize { self.t.dealer }
    #[getter]
    fn round_wind_idx(&self) -> usize { self.t.round_wind_idx }
    #[getter]
    fn round_number(&self) -> usize { self.t.round_number }
    #[getter]
    fn turn(&self) -> usize { self.t.turn }
    #[getter]
    fn points(&self) -> Vec<i64> { self.t.points.to_vec() }
    #[getter]
    fn start_points(&self) -> Vec<i64> { self.t.start_points.to_vec() }
    #[getter]
    fn kyotaku(&self) -> i64 { self.t.kyotaku }
    #[getter]
    fn wall(&self) -> Vec<String> { strs(&self.t.wall) }
    #[getter]
    fn dead_wall(&self) -> Vec<String> { strs(&self.t.dead_wall) }
    #[getter]
    fn dora_indicators(&self) -> Vec<String> { strs(&self.t.dora_indicators) }
    #[getter]
    fn ura_indicators(&self) -> Vec<String> { strs(&self.t.ura_indicators) }
    #[getter]
    fn hands(&self) -> Vec<Vec<String>> { (0..4).map(|p| strs(&self.t.hands[p])).collect() }
    #[getter]
    fn red(&self) -> Vec<Vec<u8>> { (0..4).map(|p| self.t.red[p].to_vec()).collect() }
    #[getter]
    fn last_drawn(&self) -> Vec<Option<String>> {
        (0..4).map(|p| self.t.last_drawn[p].map(to_string)).collect()
    }
    #[getter]
    fn last_drawn_red(&self) -> Vec<bool> { self.t.last_drawn_red.to_vec() }
    #[getter]
    fn finished(&self) -> bool { self.t.finished }
}
