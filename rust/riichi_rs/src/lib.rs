//! riichi_rs — Rust port of the LLM_Mahjong single-deal engine (PyMahjongTable),
//! the v1r observation encoder and a vectorized self-play environment.
//! Parity with the Python engine is the contract; see tests/test_rust_parity.py.
use pyo3::prelude::*;

pub mod pyrandom;
pub mod tiles;
pub mod shanten;
pub mod table;
pub mod pytable;

fn parse_tiles(tiles: Vec<String>) -> PyResult<Vec<tiles::Tile>> {
    tiles.iter().map(|s| tiles::parse(s).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("bad tile {s}")))).collect()
}

/// PyMahjongTable._shanten(tiles, num_melds)
#[pyfunction]
#[pyo3(name = "shanten")]
fn py_shanten(tiles: Vec<String>, num_melds: usize) -> PyResult<i32> {
    Ok(shanten::shanten_hand(&parse_tiles(tiles)?, num_melds))
}

/// PyMahjongTable._waits_of(tiles, n_melds) as a sorted list of tile strings
#[pyfunction]
#[pyo3(name = "waits")]
fn py_waits(tiles: Vec<String>, n_melds: usize) -> PyResult<Vec<String>> {
    Ok(shanten::waits_of(&parse_tiles(tiles)?, n_melds).into_iter().map(tiles::to_string).collect())
}

#[pyfunction]
#[pyo3(name = "tile_only_as_triplet")]
fn py_tile_only_as_triplet(tiles: Vec<String>, tile: String, n_sets: i32) -> PyResult<bool> {
    let t = tiles::parse(&tile).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad tile"))?;
    Ok(shanten::tile_only_as_triplet(&parse_tiles(tiles)?, t, n_sets))
}

#[pymodule]
fn riichi_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyrandom::PyRandom>()?;
    m.add_class::<pytable::PyTable>()?;
    m.add_function(wrap_pyfunction!(py_shanten, m)?)?;
    m.add_function(wrap_pyfunction!(py_waits, m)?)?;
    m.add_function(wrap_pyfunction!(py_tile_only_as_triplet, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
