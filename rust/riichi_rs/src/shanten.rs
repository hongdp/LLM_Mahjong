//! Shanten / waits — a verbatim port of the `mahjong` 2.0.0 library's
//! algorithm (regular + chiitoitsu + kokushi) plus the Python engine's
//! meld padding, wait-candidate pruning and riichi-ankan decomposition check.
use crate::tiles::*;

pub const AGARI: i32 = -1;

pub type Counts = [u8; 34];

pub fn counts_of(tiles: &[Tile]) -> Counts {
    let mut c = [0u8; 34];
    for &t in tiles {
        c[norm(t) as usize] += 1;
    }
    c
}

const TERMINAL_HONOR: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

pub fn shanten_chiitoitsu(c: &Counts) -> i32 {
    let pairs = c.iter().filter(|&&x| x >= 2).count() as i32;
    if pairs == 7 {
        return AGARI;
    }
    let kinds = c.iter().filter(|&&x| x >= 1).count() as i32;
    6 - pairs + if kinds < 7 { 7 - kinds } else { 0 }
}

pub fn shanten_kokushi(c: &Counts) -> i32 {
    let mut completed = 0;
    let mut terminals = 0;
    for &i in TERMINAL_HONOR.iter() {
        if c[i] >= 2 {
            completed += 1;
        }
        if c[i] != 0 {
            terminals += 1;
        }
    }
    13 - terminals - if completed > 0 { 1 } else { 0 }
}

struct Regular {
    tiles: [i32; 34],
    melds: i32,
    tatsu: i32,
    pairs: i32,
    jidahai: i32,
    four: u32,
    isolated: u32,
    min: i32,
}

impl Regular {
    fn calculate(c: &Counts) -> Result<i32, ()> {
        let n: i32 = c.iter().map(|&x| x as i32).sum();
        if n > 14 || n % 3 == 0 {
            return Err(());
        }
        let mut r = Regular {
            tiles: [0; 34],
            melds: 0,
            tatsu: 0,
            pairs: 0,
            jidahai: 0,
            four: 0,
            isolated: 0,
            min: 8,
        };
        for i in 0..34 {
            r.tiles[i] = c[i] as i32;
        }
        r.remove_character_tiles(n);
        let init_mentsu = (14 - n) / 3;
        for i in 0..27 {
            r.four |= ((r.tiles[i] == 4) as u32) << i;
        }
        r.melds += init_mentsu;
        r.run(0);
        Ok(r.min)
    }

    fn remove_character_tiles(&mut self, nc: i32) {
        let mut four = 0u32;
        let mut isolated = 0u32;
        for i in 27..34 {
            match self.tiles[i] {
                4 => {
                    self.melds += 1;
                    self.jidahai += 1;
                    four |= 1 << (i - 27);
                    isolated |= 1 << (i - 27);
                }
                3 => self.melds += 1,
                2 => self.pairs += 1,
                1 => isolated |= 1 << (i - 27),
                _ => {}
            }
        }
        if self.jidahai > 0 && nc % 3 == 2 {
            self.jidahai -= 1;
        }
        if isolated != 0 {
            self.isolated |= 1 << 27;
            if (four | isolated) == four {
                self.four |= 1 << 27;
            }
        }
    }

    fn update_result(&mut self) {
        let mut ret = 8 - self.melds * 2 - self.tatsu - self.pairs;
        let mut kouho = self.melds + self.tatsu;
        if self.pairs > 0 {
            kouho += self.pairs - 1;
        } else if self.four != 0
            && self.isolated != 0
            && (self.four | self.isolated) == self.four
        {
            ret += 1;
        }
        if kouho > 4 {
            ret += kouho - 4;
        }
        if ret != AGARI && ret < self.jidahai {
            ret = self.jidahai;
        }
        if ret < self.min {
            self.min = ret;
        }
    }

    #[inline]
    fn inc_set(&mut self, k: usize) { self.tiles[k] -= 3; self.melds += 1; }
    #[inline]
    fn dec_set(&mut self, k: usize) { self.tiles[k] += 3; self.melds -= 1; }
    #[inline]
    fn inc_pair(&mut self, k: usize) { self.tiles[k] -= 2; self.pairs += 1; }
    #[inline]
    fn dec_pair(&mut self, k: usize) { self.tiles[k] += 2; self.pairs -= 1; }
    #[inline]
    fn inc_syuntsu(&mut self, k: usize) { self.tiles[k] -= 1; self.tiles[k + 1] -= 1; self.tiles[k + 2] -= 1; self.melds += 1; }
    #[inline]
    fn dec_syuntsu(&mut self, k: usize) { self.tiles[k] += 1; self.tiles[k + 1] += 1; self.tiles[k + 2] += 1; self.melds -= 1; }
    #[inline]
    fn inc_tatsu1(&mut self, k: usize) { self.tiles[k] -= 1; self.tiles[k + 1] -= 1; self.tatsu += 1; }
    #[inline]
    fn dec_tatsu1(&mut self, k: usize) { self.tiles[k] += 1; self.tiles[k + 1] += 1; self.tatsu -= 1; }
    #[inline]
    fn inc_tatsu2(&mut self, k: usize) { self.tiles[k] -= 1; self.tiles[k + 2] -= 1; self.tatsu += 1; }
    #[inline]
    fn dec_tatsu2(&mut self, k: usize) { self.tiles[k] += 1; self.tiles[k + 2] += 1; self.tatsu -= 1; }
    #[inline]
    fn inc_iso(&mut self, k: usize) { self.tiles[k] -= 1; self.isolated |= 1 << k; }
    #[inline]
    fn dec_iso(&mut self, k: usize) { self.tiles[k] += 1; self.isolated &= !(1u32 << k); }

    fn run(&mut self, mut depth: usize) {
        if self.min == AGARI {
            return;
        }
        while depth < 27 && self.tiles[depth] == 0 {
            depth += 1;
        }
        if depth >= 27 {
            self.update_result();
            return;
        }
        let i = depth % 9;
        let d = depth;
        let t = &self.tiles;
        if t[d] == 4 {
            self.inc_set(d);
            if i < 7 && self.tiles[d + 2] > 0 {
                if self.tiles[d + 1] > 0 {
                    self.inc_syuntsu(d);
                    self.run(d + 1);
                    self.dec_syuntsu(d);
                }
                self.inc_tatsu2(d);
                self.run(d + 1);
                self.dec_tatsu2(d);
            }
            if i < 8 && self.tiles[d + 1] > 0 {
                self.inc_tatsu1(d);
                self.run(d + 1);
                self.dec_tatsu1(d);
            }
            self.inc_iso(d);
            self.run(d + 1);
            self.dec_iso(d);
            self.dec_set(d);
            self.inc_pair(d);
            if i < 7 && self.tiles[d + 2] > 0 {
                if self.tiles[d + 1] > 0 {
                    self.inc_syuntsu(d);
                    self.run(d);
                    self.dec_syuntsu(d);
                }
                self.inc_tatsu2(d);
                self.run(d + 1);
                self.dec_tatsu2(d);
            }
            if i < 8 && self.tiles[d + 1] > 0 {
                self.inc_tatsu1(d);
                self.run(d + 1);
                self.dec_tatsu1(d);
            }
            self.dec_pair(d);
        }
        if self.tiles[d] == 3 {
            self.inc_set(d);
            self.run(d + 1);
            self.dec_set(d);
            self.inc_pair(d);
            if i < 7 && self.tiles[d + 1] > 0 && self.tiles[d + 2] > 0 {
                self.inc_syuntsu(d);
                self.run(d + 1);
                self.dec_syuntsu(d);
            } else {
                if i < 7 && self.tiles[d + 2] > 0 {
                    self.inc_tatsu2(d);
                    self.run(d + 1);
                    self.dec_tatsu2(d);
                }
                if i < 8 && self.tiles[d + 1] > 0 {
                    self.inc_tatsu1(d);
                    self.run(d + 1);
                    self.dec_tatsu1(d);
                }
            }
            self.dec_pair(d);
            if i < 7 && self.tiles[d + 2] >= 2 && self.tiles[d + 1] >= 2 {
                self.inc_syuntsu(d);
                self.inc_syuntsu(d);
                self.run(d);
                self.dec_syuntsu(d);
                self.dec_syuntsu(d);
            }
        }
        if self.tiles[d] == 2 {
            self.inc_pair(d);
            self.run(d + 1);
            self.dec_pair(d);
            if i < 7 && self.tiles[d + 2] > 0 && self.tiles[d + 1] > 0 {
                self.inc_syuntsu(d);
                self.run(d);
                self.dec_syuntsu(d);
            }
        }
        if self.tiles[d] == 1 {
            if i < 6 && self.tiles[d + 1] == 1 && self.tiles[d + 2] > 0 && self.tiles[d + 3] != 4 {
                self.inc_syuntsu(d);
                self.run(d + 2);
                self.dec_syuntsu(d);
            } else {
                self.inc_iso(d);
                self.run(d + 1);
                self.dec_iso(d);
                if i < 7 && self.tiles[d + 2] > 0 {
                    if self.tiles[d + 1] > 0 {
                        self.inc_syuntsu(d);
                        self.run(d + 1);
                        self.dec_syuntsu(d);
                    }
                    self.inc_tatsu2(d);
                    self.run(d + 1);
                    self.dec_tatsu2(d);
                }
                if i < 8 && self.tiles[d + 1] > 0 {
                    self.inc_tatsu1(d);
                    self.run(d + 1);
                    self.dec_tatsu1(d);
                }
            }
        }
    }
}

/// `Shanten.calculate_shanten(tiles_34)` with chiitoitsu and kokushi enabled.
/// Err(()) mirrors the library's ValueError on invalid tile counts.
pub fn shanten_counts(c: &Counts) -> Result<i32, ()> {
    let r = Regular::calculate(c)?;
    let ch = shanten_chiitoitsu(c);
    let ko = shanten_kokushi(c);
    Ok(r.min(ch).min(ko))
}

/// Python `pad_for_melds`: one dummy triplet per meld, using tile types
/// absent from the hand, taken from 33 downwards.
pub fn padded_counts(tiles: &[Tile], num_melds: usize) -> Counts {
    let mut c = counts_of(tiles);
    if num_melds == 0 {
        return c;
    }
    let mut m = 0usize;
    for i in (0..34).rev() {
        if m >= num_melds {
            break;
        }
        if c[i] == 0 {
            c[i] = 3;
            m += 1;
        }
    }
    c
}

/// PyMahjongTable._shanten: clamp the meld count so the padded hand never
/// exceeds 14 tiles; invalid counts report 8 ("far from tenpai").
pub fn shanten_hand(tiles: &[Tile], num_melds: usize) -> i32 {
    let n = tiles.len();
    let nm = if n > 14 { 0 } else { num_melds.min((14 - n) / 3) };
    let c = padded_counts(tiles, nm);
    shanten_counts(&c).unwrap_or(8)
}

const ORPHANS: [bool; 34] = {
    let mut a = [false; 34];
    let idx = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut k = 0;
    while k < idx.len() {
        a[idx[k]] = true;
        k += 1;
    }
    a
};

/// Python `_wait_candidates`: sorted 34-indices that could complete `tiles`.
pub fn wait_candidates(tiles: &[Tile]) -> Vec<u8> {
    let idx: Vec<u8> = tiles.iter().map(|&t| norm(t)).collect();
    if idx.iter().all(|&i| ORPHANS[i as usize]) {
        return (0..34).collect();
    }
    let mut mask = [false; 34];
    for &i in &idx {
        mask[i as usize] = true;
        if i < 27 {
            let r = (i % 9) as i32;
            let base = i - (i % 9);
            for d in [-2i32, -1, 1, 2] {
                let rr = r + d;
                if (0..=8).contains(&rr) {
                    mask[(base as i32 + rr) as usize] = true;
                }
            }
        }
    }
    (0..34u8).filter(|&i| mask[i as usize]).collect()
}

/// PyMahjongTable._waits_of: tiles completing a 13-tile-state hand (empty
/// unless the hand is tenpai). Returned in candidate (34-index) order.
pub fn waits_of(tiles: &[Tile], n_melds: usize) -> Vec<Tile> {
    if shanten_hand(tiles, n_melds) != 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut with = tiles.to_vec();
    for i in wait_candidates(tiles) {
        if tiles.iter().filter(|&&t| norm(t) == i).count() >= 4 {
            continue;
        }
        with.push(i);
        if shanten_hand(&with, n_melds) == AGARI {
            out.push(i);
        }
        with.pop();
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    Pair(u8),
    Set(u8),
    Seq(u8),
}

/// Python `_decompositions`: every standard reading of the counts as
/// `n_sets` sets plus one pair.
pub fn decompositions(c: &Counts, n_sets: i32) -> Vec<Vec<Piece>> {
    fn rec(counts: &mut [i32; 34], mut i: usize, sets_left: i32, pair_used: bool,
           acc: &mut Vec<Piece>, out: &mut Vec<Vec<Piece>>) {
        while i < 34 && counts[i] == 0 {
            i += 1;
        }
        if i == 34 {
            if sets_left == 0 && pair_used {
                out.push(acc.clone());
            }
            return;
        }
        let cnt = counts[i];
        if !pair_used && cnt >= 2 {
            counts[i] -= 2;
            acc.push(Piece::Pair(i as u8));
            rec(counts, i, sets_left, true, acc, out);
            acc.pop();
            counts[i] += 2;
        }
        if sets_left == 0 {
            return;
        }
        if cnt >= 3 {
            counts[i] -= 3;
            acc.push(Piece::Set(i as u8));
            rec(counts, i, sets_left - 1, pair_used, acc, out);
            acc.pop();
            counts[i] += 3;
        }
        if i < 27 && i % 9 <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0 {
            counts[i] -= 1; counts[i + 1] -= 1; counts[i + 2] -= 1;
            acc.push(Piece::Seq(i as u8));
            rec(counts, i, sets_left - 1, pair_used, acc, out);
            acc.pop();
            counts[i] += 1; counts[i + 1] += 1; counts[i + 2] += 1;
        }
    }
    let mut counts = [0i32; 34];
    for i in 0..34 {
        counts[i] = c[i] as i32;
    }
    let mut out = Vec::new();
    let mut acc = Vec::new();
    rec(&mut counts, 0, n_sets, false, &mut acc, &mut out);
    out
}

/// Python `_tile_only_as_triplet` (RCR 3.12 (2)).
pub fn tile_only_as_triplet(tiles: &[Tile], tile: Tile, n_sets: i32) -> bool {
    let c = counts_of(tiles);
    let k = norm(tile);
    let mut found = false;
    for reading in decompositions(&c, n_sets) {
        found = true;
        if reading.contains(&Piece::Pair(k)) {
            return false;
        }
        let has_set = reading.contains(&Piece::Set(k));
        if !has_set
            && reading.iter().any(|p| matches!(p, Piece::Seq(i) if *i <= k && k <= *i + 2))
        {
            return false;
        }
    }
    found
}
