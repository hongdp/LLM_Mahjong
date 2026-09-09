//! riichi_rs — Rust port of the LLM_Mahjong single-deal engine (PyMahjongTable),
//! the v1r observation encoder and a vectorized self-play environment.
//! Parity with the Python engine is the contract; see tests/test_rust_parity.py.
use pyo3::prelude::*;

pub mod pyrandom;
pub mod tiles;
pub mod table;
pub mod pytable;

#[pymodule]
fn riichi_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyrandom::PyRandom>()?;
    m.add_class::<pytable::PyTable>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
