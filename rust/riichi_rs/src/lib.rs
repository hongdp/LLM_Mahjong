//! riichi_rs — Rust port of the LLM_Mahjong single-deal engine (PyMahjongTable),
//! the v1r observation encoder and a vectorized self-play environment.
//! Parity with the Python engine is the contract; see tests/test_rust_parity.py.
use pyo3::prelude::*;

pub mod pyrandom;
pub mod tiles;
pub mod shanten;
pub mod score;
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

/// Differential-test entry: HandCalculator.estimate_hand_value as the engine calls it.
/// melds: [(type, [tiles], opened)] with type in chi/pon/kan/ankan/shouminkan.
#[pyfunction]
#[pyo3(name = "estimate_hand")]
#[allow(clippy::too_many_arguments)]
fn py_estimate_hand(py: Python<'_>, tiles: Vec<String>, win_tile: String, melds: Vec<(String, Vec<String>, bool)>,
                    dora: Vec<String>, ura: Vec<String>, aka: u32, cfg: std::collections::HashMap<String, i64>)
                    -> PyResult<Option<PyObject>> {
    use pyo3::types::PyDict;
    let all = parse_tiles(tiles)?;
    let mut counts = [0u8; 34];
    for &t in &all { counts[tiles::norm(t) as usize] += 1; }
    let win = tiles::norm(tiles::parse(&win_tile).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad win tile"))?);
    let mut ms = Vec::new();
    for (ty, ts, opened) in melds {
        let mtype = match ty.as_str() {
            "chi" => score::MeldType::Chi, "pon" => score::MeldType::Pon,
            "kan" | "ankan" => score::MeldType::Kan, "shouminkan" => score::MeldType::Shouminkan,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("bad meld type")),
        };
        let t34: Vec<u8> = parse_tiles(ts)?.into_iter().map(tiles::norm).collect();
        ms.push(score::ScoreMeld { mtype, tiles: t34, opened });
    }
    let d: Vec<u8> = parse_tiles(dora)?.into_iter().map(tiles::norm).collect();
    let u: Vec<u8> = parse_tiles(ura)?.into_iter().map(tiles::norm).collect();
    let g = |k: &str| cfg.get(k).copied().unwrap_or(0) != 0;
    let c = score::Config {
        is_tsumo: g("is_tsumo"), is_riichi: g("is_riichi"), is_daburu_riichi: g("is_daburu_riichi"),
        is_ippatsu: g("is_ippatsu"), is_rinshan: g("is_rinshan"), is_chankan: g("is_chankan"),
        is_haitei: g("is_haitei"), is_houtei: g("is_houtei"), is_tenhou: g("is_tenhou"), is_chiihou: g("is_chiihou"),
        player_wind: cfg.get("player_wind").copied().unwrap_or(27) as u8,
        round_wind: cfg.get("round_wind").copied().unwrap_or(27) as u8,
    };
    let inp = score::HandInput { tiles_34: counts, win_tile: win, melds: &ms, dora_indicators: &d, ura_indicators: &u, aka_count: aka, cfg: c };
    match score::estimate(&inp) {
        None => Ok(None),
        Some(r) => {
            let dct = PyDict::new_bound(py);
            dct.set_item("han", r.han)?;
            dct.set_item("fu", r.fu)?;
            dct.set_item("main", r.main)?;
            dct.set_item("additional", r.additional)?;
            dct.set_item("yaku", r.yaku.iter().map(|y| y.name()).collect::<Vec<String>>())?;
            Ok(Some(dct.into()))
        }
    }
}

#[pymodule]
fn riichi_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyrandom::PyRandom>()?;
    m.add_class::<pytable::PyTable>()?;
    m.add_function(wrap_pyfunction!(py_shanten, m)?)?;
    m.add_function(wrap_pyfunction!(py_waits, m)?)?;
    m.add_function(wrap_pyfunction!(py_tile_only_as_triplet, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_hand, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
