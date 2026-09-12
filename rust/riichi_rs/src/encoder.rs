//! v1r observation encoder (21 x 34 planes + 20 scalars) and the native
//! 374-slot action index — bit-identical to src/agents/dnn/encoder.py.
use crate::table::Table;
use crate::tiles::*;

pub const N_PLANES: usize = 21;
pub const N_SCALARS: usize = 20;
pub const ACTION_DIM: usize = 374;
/// Plane quantisation step: every plane value the encoders emit lies on the k/20 grid
/// (binary planes and the v3r river-order plane (j+1)/20 clipped at 1), so planes travel
/// as u8 = round(v*20) and are widened on the training device with `u8.float() / 20`
/// (f32 division reproduces the f64-then-f32 encoder values bit-exactly; see
/// tests/test_rust_encoder_parity.py::test_plane_quantisation_exact).
pub const PLANE_Q: f32 = 20.0;

pub fn quantize_planes(f: &[f32], out: &mut [u8]) {
    debug_assert!(f.len() == out.len());
    for (o, &v) in out.iter_mut().zip(f.iter()) {
        *o = (v * PLANE_Q).round() as u8;
    }
}

fn type_id(kind: &str) -> Option<usize> {
    Some(match kind {
        "discard" => 0, "riichi" => 1, "chi" => 2, "pon" => 3, "kan" => 4, "ron" => 5, "tsumo" => 6,
        "skip" => 7, "discard0" => 8, "riichi0" => 9, "kyuushu" => 10, _ => return None,
    })
}

/// encoder.action_to_index
pub fn action_to_index(xml: &str) -> Option<usize> {
    let a = crate::table::parse_action(xml)?;
    let mut kind = a.kind.clone();
    if (kind == "discard" || kind == "riichi") && a.tile.as_deref().map(|t| t.starts_with('0')).unwrap_or(false) {
        kind.push('0');
    }
    let tid = type_id(&kind)?;
    let key = if kind == "chi" && a.with.is_some() {
        let first = a.with.as_deref().unwrap().split_whitespace().next()?;
        norm(parse(first)?) as usize
    } else if let Some(t) = a.tile.as_deref() {
        norm(parse(t)?) as usize
    } else {
        0
    };
    Some(tid * 34 + key)
}

/// encoder.legal_mask: mask over ACTION_DIM + slot -> action lookup (first action wins a slot).
pub fn legal_mask(actions: &[String]) -> ([bool; ACTION_DIM], Vec<(usize, usize)>) {
    let mut mask = [false; ACTION_DIM];
    let mut lookup = Vec::with_capacity(actions.len());     // (slot, action index)
    for (i, a) in actions.iter().enumerate() {
        if let Some(idx) = action_to_index(a) {
            if mask[idx] {
                continue;
            }
            mask[idx] = true;
            lookup.push((idx, i));
        }
    }
    (mask, lookup)
}

/// Fill `planes` (N_PLANES*34 row-major) and `scalars` (N_SCALARS) for `pid`.
pub fn encode_v1r(t: &Table, pid: usize, planes: &mut [f32], scalars: &mut [f32]) {
    debug_assert!(planes.len() == N_PLANES * 34 && scalars.len() == N_SCALARS);
    for v in planes.iter_mut() {
        *v = 0.0;
    }
    for v in scalars.iter_mut() {
        *v = 0.0;
    }
    let row = |p: usize| p * 34;
    // 0-3: hand count >= k+1
    let mut counts = [0u8; 34];
    for &h in &t.hands[pid] {
        counts[norm(h) as usize] += 1;
    }
    for k in 0..4 {
        for i in 0..34 {
            if counts[i] as usize >= k + 1 {
                planes[row(k) + i] = 1.0;
            }
        }
    }
    // 4-7: melds presence (own, then opponents by offset)
    for off in 0..4 {
        let p = (pid + off) % 4;
        for m in &t.melds[p] {
            for &x in &m.tiles {
                planes[row(4 + off) + norm(x) as usize] = 1.0;
            }
        }
    }
    // 8-11: rivers presence
    for off in 0..4 {
        let p = (pid + off) % 4;
        for &(x, _) in &t.discards[p] {
            planes[row(8 + off) + norm(x) as usize] = 1.0;
        }
    }
    // 12: dora tiles
    for &ind in &t.dora_indicators {
        planes[row(12) + dora_from_indicator(ind) as usize] = 1.0;
    }
    // 13: last discard
    if let Some(ld) = t.last_discard {
        planes[row(13) + norm(ld) as usize] = 1.0;
    }
    // 14: own furiten river
    for &x in &t.furiten_river[pid] {
        planes[row(14) + norm(x) as usize] = 1.0;
    }
    // 15-20: red planes
    let r = |k: usize| row(15 + k);
    planes[r(5) + 27 + t.round_wind_idx] = 1.0;
    planes[r(5) + 27 + (pid + 4 - t.dealer) % 4] = 1.0;
    for i in 31..34 {
        planes[r(5) + i] = 1.0;
    }
    for off in 0..4 {
        let p = (pid + off) % 4;
        if off == 0 {
            for s in 0..3 {
                if t.red[p][s] > 0 {
                    planes[r(0) + s * 9 + 4] = 1.0;
                }
            }
        }
        for &(x, _) in &t.discards[p] {
            if is_red(x) {
                planes[r(1 + off) + norm(x) as usize] = 1.0;
            }
        }
        for m in &t.melds[p] {
            if m.red > 0 {
                planes[r(1 + off) + norm(m.tiles[0]) as usize] = 1.0;
            }
        }
    }
    // scalars
    for off in 0..4 {
        let p = (pid + off) % 4;
        scalars[off] = ((t.points[p] - 25000) as f64 / 25000.0) as f32;
        scalars[4 + off] = if t.riichi[p] { 1.0 } else { 0.0 };
    }
    scalars[8] = (t.wall.len() as f64 / 70.0) as f32;
    scalars[9] = (t.kyotaku as f64 / 4.0) as f32;
    scalars[10] = (t.hands[pid].len() as f64 / 14.0) as f32;
    scalars[11] = (t.melds[pid].len() as f64 / 4.0) as f32;
    if t.round_wind_idx >= 2 {
        scalars[12] = 1.0;
        scalars[13] = 1.0;
    } else {
        scalars[12 + t.round_wind_idx] = 1.0;
    }
    scalars[14 + (pid + 4 - t.dealer) % 4] = 1.0;
    scalars[18] = if t.turn == pid { 1.0 } else { 0.0 };
    scalars[19] = if t.last_discarder == Some((pid + 3) % 4) { 1.0 } else { 0.0 };
}

/// selfplay.potential: -2 * best reachable shanten (memoised in the Python engine; cheap here).
pub fn potential(t: &Table, pid: usize) -> f64 {
    let tiles = &t.hands[pid];
    let nm = t.melds[pid].len();
    let sh = if tiles.len() % 3 == 2 {
        let mut best: Option<i32> = None;
        let mut seen = [false; 34];
        for &x in tiles {
            if seen[x as usize] {
                continue;
            }
            seen[x as usize] = true;
            let mut h = tiles.clone();
            let pos = h.iter().position(|&y| y == x).unwrap();
            h.remove(pos);
            let s = t.shanten_of(&h, nm);
            if best.map_or(true, |b| s < b) {
                best = Some(s);
                if s <= 0 {
                    break;
                }
            }
        }
        best.unwrap_or(8)
    } else {
        t.shanten_of(tiles, nm)
    };
    -2.0 * sh as f64
}

// ======================================================================
// v3r: the complete public record (50 planes) + 6 red planes = 56, 29 scalars
// (encoder._encode_v3 + _red_planes), bit-identical to the Python encoder.
// ======================================================================
pub const N_PLANES_V3R: usize = 56;
pub const N_SCALARS_V3: usize = 29;

pub fn planes_of(variant: &str) -> usize {
    match variant {
        "v3r" => N_PLANES_V3R,
        _ => N_PLANES,
    }
}

pub fn scalars_of(variant: &str) -> usize {
    match variant {
        "v3r" => N_SCALARS_V3,
        _ => N_SCALARS,
    }
}

fn red_planes(t: &Table, pid: usize, planes: &mut [f32], base_plane: usize) {
    let r = |k: usize| (base_plane + k) * 34;
    planes[r(5) + 27 + t.round_wind_idx] = 1.0;
    planes[r(5) + 27 + (pid + 4 - t.dealer) % 4] = 1.0;
    for i in 31..34 {
        planes[r(5) + i] = 1.0;
    }
    for off in 0..4 {
        let p = (pid + off) % 4;
        if off == 0 {
            for s in 0..3 {
                if t.red[p][s] > 0 {
                    planes[r(0) + s * 9 + 4] = 1.0;
                }
            }
        }
        for &(x, _) in &t.discards[p] {
            if is_red(x) {
                planes[r(1 + off) + norm(x) as usize] = 1.0;
            }
        }
        for m in &t.melds[p] {
            if m.red > 0 {
                planes[r(1 + off) + norm(m.tiles[0]) as usize] = 1.0;
            }
        }
    }
}

pub fn encode_v3r(t: &Table, pid: usize, planes: &mut [f32], scalars: &mut [f32]) {
    debug_assert!(planes.len() == N_PLANES_V3R * 34 && scalars.len() == N_SCALARS_V3);
    for v in planes.iter_mut() {
        *v = 0.0;
    }
    for v in scalars.iter_mut() {
        *v = 0.0;
    }
    let row = |p: usize| p * 34;
    let mut counts = [0f32; 34];
    for &h in &t.hands[pid] {
        counts[norm(h) as usize] += 1.0;
    }
    for k in 0..4 {
        for i in 0..34 {
            if counts[i] >= (k + 1) as f32 {
                planes[row(k) + i] = 1.0;
            }
        }
    }
    let mut seen = counts;
    for off in 0..4 {
        let p = (pid + off) % 4;
        for m in &t.melds[p] {
            for &x in &m.tiles {
                planes[row(4 + off) + norm(x) as usize] = 1.0;
                seen[norm(x) as usize] += 1.0;
            }
        }
        for &(x, _) in &t.discards[p] {
            planes[row(8 + off) + norm(x) as usize] = 1.0;
        }
        for &x in &t.furiten_river[p] {
            seen[norm(x) as usize] += 1.0;
        }
    }
    for &ind in &t.dora_indicators {
        planes[row(12) + dora_from_indicator(ind) as usize] = 1.0;
        seen[norm(ind) as usize] += 1.0;
    }
    if let Some(ld) = t.last_discard {
        planes[row(13) + norm(ld) as usize] = 1.0;
    }
    for &x in &t.furiten_river[pid] {
        planes[row(14) + norm(x) as usize] = 1.0;
    }
    // 15..30: per-seat river facts
    for off in 0..4 {
        let p = (pid + off) % 4;
        let base = 15 + 4 * off;
        for (j, e) in t.river_events[p].iter().enumerate() {
            let x = norm(e.tile) as usize;
            planes[row(base) + x] = (((j + 1) as f64 / 20.0).min(1.0)) as f32;
            if e.tsumogiri {
                planes[row(base + 1) + x] = 1.0;
            }
            if e.riichi {
                planes[row(base + 2) + x] = 1.0;
            }
            if e.called {
                planes[row(base + 3) + x] = 1.0;
            }
        }
    }
    // 31..42 meld types; 43..45 opponents' melds fed by me
    for off in 0..4 {
        let p = (pid + off) % 4;
        for m in &t.melds[p] {
            let kind = match m.kind {
                crate::table::MeldKind::Chi => 0,
                crate::table::MeldKind::Pon => 1,
                _ => 2,
            };
            for &x in &m.tiles {
                planes[row(31 + 3 * off + kind) + norm(x) as usize] = 1.0;
            }
            if off > 0 && m.from == Some(pid) {
                for &x in &m.tiles {
                    planes[row(43 + off - 1) + norm(x) as usize] = 1.0;
                }
            }
        }
    }
    for k in 0..4 {
        for i in 0..34 {
            if seen[i] >= (k + 1) as f32 {
                planes[row(46 + k) + i] = 1.0;
            }
        }
    }
    red_planes(t, pid, planes, 50);
    // scalars
    for off in 0..4 {
        let p = (pid + off) % 4;
        scalars[off] = ((t.points[p] - 25000) as f64 / 25000.0) as f32;
        scalars[4 + off] = if t.riichi[p] { 1.0 } else { 0.0 };
    }
    scalars[8] = (t.wall.len() as f64 / 70.0) as f32;
    scalars[9] = (t.kyotaku as f64 / 4.0) as f32;
    scalars[10] = (t.hands[pid].len() as f64 / 14.0) as f32;
    scalars[11] = (t.melds[pid].len() as f64 / 4.0) as f32;
    if t.round_wind_idx >= 2 {
        scalars[12] = 1.0;
        scalars[13] = 1.0;
    } else {
        scalars[12 + t.round_wind_idx] = 1.0;
    }
    scalars[14 + (pid + 4 - t.dealer) % 4] = 1.0;
    scalars[18] = if t.turn == pid { 1.0 } else { 0.0 };
    scalars[19] = if t.last_discarder == Some((pid + 3) % 4) { 1.0 } else { 0.0 };
    let mut total = 0usize;
    for off in 0..4 {
        let p = (pid + off) % 4;
        scalars[20 + off] = match t.riichi_turn[p] {
            None => 0.0,
            Some(rt) => (((rt + 1) as f64 / 20.0).min(1.0)) as f32,
        };
        scalars[24 + off] = ((t.discard_count[p] as f64 / 20.0).min(1.0)) as f32;
        total += t.discard_count[p];
    }
    scalars[28] = ((total as f64 / 70.0).min(1.0)) as f32;
}

/// Dispatch on the encoder variant name ("v1r" | "v3r").
pub fn encode(t: &Table, pid: usize, variant: &str, planes: &mut [f32], scalars: &mut [f32]) {
    match variant {
        "v3r" => encode_v3r(t, pid, planes, scalars),
        _ => encode_v1r(t, pid, planes, scalars),
    }
}
