//! Winning-hand evaluation: a port of the `mahjong` 2.0.0 library's
//! HandDivider / FuCalculator / yaku list / ScoresCalculator as configured by
//! the Python engine (open tanyao, aka dora, double yakuman, kazoe = yakuman,
//! renhou off, no honba/kyoutaku inside the calculator).
use crate::tiles::*;

pub const EAST: u8 = 27;
pub const SOUTH: u8 = 28;
pub const WEST: u8 = 29;
pub const NORTH: u8 = 30;
pub const HAKU: u8 = 31;
pub const HATSU: u8 = 32;
pub const CHUN: u8 = 33;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeldType {
    Chi,
    Pon,
    Kan,        // library Meld.KAN (open daiminkan or closed ankan)
    Shouminkan,
}

#[derive(Clone, Debug)]
pub struct ScoreMeld {
    pub mtype: MeldType,
    pub tiles: Vec<u8>,   // 34-indices in the engine's order (chi ascending)
    pub opened: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    pub is_tsumo: bool,
    pub is_riichi: bool,
    pub is_daburu_riichi: bool,
    pub is_ippatsu: bool,
    pub is_rinshan: bool,
    pub is_chankan: bool,
    pub is_haitei: bool,
    pub is_houtei: bool,
    pub is_tenhou: bool,
    pub is_chiihou: bool,
    pub player_wind: u8,
    pub round_wind: u8,
}

impl Config {
    #[inline]
    pub fn is_dealer(&self) -> bool {
        self.player_wind == EAST
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Yaku {
    Tsumo, Pinfu, Chiitoitsu, Tanyao, Riichi, DaburuRiichi, Ippatsu, Rinshan, Chankan, Haitei, Houtei,
    Tenhou, Chiihou, Chinitsu, Honitsu, Tsuisou, Honroto, Chinroto, Ryuisou, Chantai, Junchan, Ittsu,
    Ryanpeiko, Iipeiko, Sanshoku, Toitoi, Sanankou, SanshokuDouko, Shosangen, Haku, Hatsu, Chun,
    SeatWind(u8), RoundWind(u8), Daisangen, Shosuushi, Daisuushi, Chuuren, DaburuChuuren, Suuankou,
    SuuankouTanki, Sankantsu, Suukantsu, Kokushi, DaburuKokushi, Dora(u32), AkaDora(u32), UraDora(u32),
}

impl Yaku {
    pub fn name(&self) -> String {
        use Yaku::*;
        match self {
            Tsumo => "Menzen Tsumo".into(), Pinfu => "Pinfu".into(), Chiitoitsu => "Chiitoitsu".into(),
            Tanyao => "Tanyao".into(), Riichi => "Riichi".into(), DaburuRiichi => "Double Riichi".into(),
            Ippatsu => "Ippatsu".into(), Rinshan => "Rinshan Kaihou".into(), Chankan => "Chankan".into(),
            Haitei => "Haitei Raoyue".into(), Houtei => "Houtei Raoyui".into(), Tenhou => "Tenhou".into(),
            Chiihou => "Chiihou".into(), Chinitsu => "Chinitsu".into(), Honitsu => "Honitsu".into(),
            Tsuisou => "Tsuu Iisou".into(), Honroto => "Honroutou".into(), Chinroto => "Chinroutou".into(),
            Ryuisou => "Ryuuiisou".into(), Chantai => "Chantai".into(), Junchan => "Junchan".into(),
            Ittsu => "Ittsu".into(), Ryanpeiko => "Ryanpeikou".into(), Iipeiko => "Iipeiko".into(),
            Sanshoku => "Sanshoku Doujun".into(), Toitoi => "Toitoi".into(), Sanankou => "San Ankou".into(),
            SanshokuDouko => "Sanshoku Doukou".into(), Shosangen => "Shou Sangen".into(),
            Haku => "Yakuhai (haku)".into(), Hatsu => "Yakuhai (hatsu)".into(), Chun => "Yakuhai (chun)".into(),
            SeatWind(w) => format!("Yakuhai (seat wind {})", wind_name(*w)),
            RoundWind(w) => format!("Yakuhai (round wind {})", wind_name(*w)),
            Daisangen => "Daisangen".into(), Shosuushi => "Shousuushii".into(), Daisuushi => "Dai Suushii".into(),
            Chuuren => "Chuuren Poutou".into(), DaburuChuuren => "Daburu Chuuren Poutou".into(),
            Suuankou => "Suu Ankou".into(), SuuankouTanki => "Suu Ankou Tanki".into(),
            Sankantsu => "San Kantsu".into(), Suukantsu => "Suu Kantsu".into(),
            Kokushi => "Kokushi Musou".into(), DaburuKokushi => "Kokushi Musou Juusanmen Matchi".into(),
            Dora(n) => format!("Dora {n}"), AkaDora(n) => format!("Aka Dora {n}"), UraDora(n) => format!("Ura Dora {n}"),
        }
    }

    fn han(&self, open: bool) -> u32 {
        use Yaku::*;
        let (o, c): (u32, u32) = match self {
            Tsumo => (0, 1), Pinfu => (0, 1), Chiitoitsu => (0, 2), Tanyao => (1, 1), Riichi => (0, 1),
            DaburuRiichi => (0, 2), Ippatsu => (0, 1), Rinshan => (1, 1), Chankan => (1, 1), Haitei => (1, 1),
            Houtei => (1, 1), Tenhou => (0, 13), Chiihou => (0, 13), Chinitsu => (5, 6), Honitsu => (2, 3),
            Tsuisou => (13, 13), Honroto => (2, 2), Chinroto => (13, 13), Ryuisou => (13, 13), Chantai => (1, 2),
            Junchan => (2, 3), Ittsu => (1, 2), Ryanpeiko => (0, 3), Iipeiko => (0, 1), Sanshoku => (1, 2),
            Toitoi => (2, 2), Sanankou => (2, 2), SanshokuDouko => (2, 2), Shosangen => (2, 2), Haku => (1, 1),
            Hatsu => (1, 1), Chun => (1, 1), SeatWind(_) => (1, 1), RoundWind(_) => (1, 1), Daisangen => (13, 13),
            Shosuushi => (13, 13), Daisuushi => (26, 26), Chuuren => (0, 13), DaburuChuuren => (0, 26),
            Suuankou => (0, 13), SuuankouTanki => (0, 26), Sankantsu => (2, 2), Suukantsu => (13, 13),
            Kokushi => (0, 13), DaburuKokushi => (0, 26), Dora(n) => (*n, *n), AkaDora(n) => (*n, *n),
            UraDora(n) => (0, *n),
        };
        // library: `if is_open_hand and item.han_open: han_open else han_closed`
        if open && o > 0 { o } else { c }
    }

    /// library yaku_id — HandResponse sorts the yaku list by it
    fn id(&self) -> u32 {
        use Yaku::*;
        match self {
            Tsumo => 0, Riichi => 1, Ippatsu => 3, Chankan => 4, Rinshan => 5, Haitei => 6, Houtei => 7,
            DaburuRiichi => 8, Pinfu => 12, Tanyao => 13, Iipeiko => 14, Haku => 15, Hatsu => 16, Chun => 17,
            SeatWind(w) => 18 + (*w - EAST) as u32, RoundWind(w) => 22 + (*w - EAST) as u32,
            Sanshoku => 26, Ittsu => 27, Chantai => 28, Honroto => 29, Toitoi => 30, Sanankou => 31,
            Sankantsu => 32, SanshokuDouko => 33, Chiitoitsu => 34, Shosangen => 35, Honitsu => 36,
            Junchan => 37, Ryanpeiko => 38, Chinitsu => 39, Kokushi => 100, Chuuren => 101, Suuankou => 102,
            Daisangen => 103, Shosuushi => 104, Ryuisou => 105, Suukantsu => 106, Tsuisou => 107,
            Chinroto => 108, Daisuushi => 111, DaburuKokushi => 112, SuuankouTanki => 113,
            DaburuChuuren => 114, Tenhou => 115, Chiihou => 116, Dora(_) => 120, AkaDora(_) => 121,
            UraDora(_) => 122,
        }
    }

    fn is_yakuman(&self) -> bool {
        use Yaku::*;
        matches!(self, Tenhou | Chiihou | Tsuisou | Chinroto | Ryuisou | Daisangen | Shosuushi | Daisuushi
            | Chuuren | DaburuChuuren | Suuankou | SuuankouTanki | Suukantsu | Kokushi | DaburuKokushi)
    }
}

fn wind_name(w: u8) -> &'static str {
    match w {
        EAST => "east",
        SOUTH => "south",
        WEST => "west",
        _ => "north",
    }
}

#[derive(Clone, Debug)]
pub struct ScoreResult {
    pub han: u32,
    pub fu: u32,
    pub yaku: Vec<Yaku>,
    pub main: i64,
    pub additional: i64,
}

// ---------------- divider ----------------
// block = (tile34, ty) with ty: 0 quad, 1 triplet, 2 pair, 3 sequence (library _BlockType order)
type Block = (u8, u8);
const QUAD: u8 = 0;
const TRIPLET: u8 = 1;
const PAIR: u8 = 2;
const SEQ: u8 = 3;

fn block_tiles(b: Block) -> Vec<u8> {
    match b.1 {
        QUAD => vec![b.0; 4],
        TRIPLET => vec![b.0; 3],
        PAIR => vec![b.0; 2],
        _ => vec![b.0, b.0 + 1, b.0 + 2],
    }
}

fn decompose_without_pair(h: &mut [i32; 9], blocks: &mut Vec<Block>, i: usize, suit: u8, remaining: i32,
                          out: &mut Vec<Vec<Block>>) {
    if i == 9 {
        if remaining == 0 {
            out.push(blocks.clone());
        }
        return;
    }
    if h[i] == 0 {
        decompose_without_pair(h, blocks, i + 1, suit, remaining, out);
        return;
    }
    if i < 7 && h[i] >= 1 && h[i + 1] >= 1 && h[i + 2] >= 1 {
        h[i] -= 1; h[i + 1] -= 1; h[i + 2] -= 1;
        blocks.push((suit + i as u8, SEQ));
        decompose_without_pair(h, blocks, i, suit, remaining - 3, out);
        blocks.pop();
        h[i] += 1; h[i + 1] += 1; h[i + 2] += 1;
    }
    if h[i] >= 3 {
        h[i] -= 3;
        blocks.push((suit + i as u8, TRIPLET));
        decompose_without_pair(h, blocks, i + 1, suit, remaining - 3, out);
        blocks.pop();
        h[i] += 3;
    }
}

fn decompose_single_color(h: &mut [i32; 9], suit: u8) -> Vec<Vec<Block>> {
    let remaining: i32 = h.iter().sum();
    let mut out = Vec::new();
    decompose_without_pair(h, &mut Vec::new(), 0, suit, remaining, &mut out);
    if out.is_empty() {
        for pair in 0..9usize {
            if h[pair] < 2 {
                continue;
            }
            h[pair] -= 2;
            let mut blocks = vec![(suit + pair as u8, PAIR)];
            let mut comb = Vec::new();
            decompose_without_pair(h, &mut blocks, 0, suit, remaining - 2, &mut comb);
            h[pair] += 2;
            out.extend(comb);
        }
    }
    out
}

fn decompose_honors(h: &[i32]) -> Option<Vec<Block>> {
    let mut has_pair = false;
    let mut blocks = Vec::new();
    for (i, &c) in h.iter().enumerate() {
        match c {
            0 => {}
            2 => {
                if has_pair {
                    return None;
                }
                blocks.push((27 + i as u8, PAIR));
                has_pair = true;
            }
            3 => blocks.push((27 + i as u8, TRIPLET)),
            _ => return None,
        }
    }
    Some(blocks)
}

fn meld_block(m: &ScoreMeld) -> Block {
    match m.mtype {
        MeldType::Chi => (m.tiles[0], SEQ),
        MeldType::Pon => (m.tiles[0], TRIPLET),
        MeldType::Kan | MeldType::Shouminkan => (m.tiles[0], QUAD),
    }
}

/// HandDivider.divide_hand: every decomposition, each as sorted blocks;
/// decompositions sorted lexicographically (chiitoitsu takes part in the sort).
pub fn divide_hand(tiles_34: &[u8; 34], melds: &[ScoreMeld]) -> Vec<Vec<Block>> {
    if tiles_34.iter().any(|&c| c > 4) {
        return Vec::new();
    }
    let meld_blocks: Vec<Block> = melds.iter().map(meld_block).collect();
    let mut pure = [0i32; 34];
    for i in 0..34 {
        pure[i] = tiles_34[i] as i32;
    }
    for mb in &meld_blocks {
        for t in block_tiles(*mb) {
            pure[t as usize] -= 1;
        }
    }
    let mut man = [0i32; 9];
    let mut pin = [0i32; 9];
    let mut sou = [0i32; 9];
    man.copy_from_slice(&pure[0..9]);
    pin.copy_from_slice(&pure[9..18]);
    sou.copy_from_slice(&pure[18..27]);
    let man_c = decompose_single_color(&mut man, 0);
    let pin_c = decompose_single_color(&mut pin, 9);
    let sou_c = decompose_single_color(&mut sou, 18);
    let honors = decompose_honors(&pure[27..34]);
    let mut combos: Vec<Vec<Block>> = Vec::new();
    if meld_blocks.is_empty() {
        if pure.iter().all(|&c| c == 0 || c == 2) {
            let pairs: Vec<Block> = (0..34).filter(|&i| pure[i] == 2).map(|i| (i as u8, PAIR)).collect();
            if pairs.len() == 7 {
                combos.push(pairs);
            }
        }
    }
    if let Some(hon) = honors {
        for m in &man_c {
            for p in &pin_c {
                for s in &sou_c {
                    let mut all: Vec<Block> = Vec::with_capacity(5);
                    all.extend(m.iter().copied());
                    all.extend(p.iter().copied());
                    all.extend(s.iter().copied());
                    all.extend(hon.iter().copied());
                    let num_pair = all.iter().filter(|b| b.1 == PAIR).count();
                    if num_pair != 1 {
                        continue;
                    }
                    all.extend(meld_blocks.iter().copied());
                    if all.len() != 5 {
                        continue;
                    }
                    all.sort();
                    combos.push(all);
                }
            }
        }
    }
    combos.sort();
    combos
}

// ---------------- helpers on decomposed hands ----------------
#[inline]
fn is_chi(item: &[u8]) -> bool {
    item.len() == 3 && item[0] + 1 == item[1] && item[1] + 1 == item[2]
}
#[inline]
fn is_pon_or_kan(item: &[u8]) -> bool {
    item.len() == 4 || (item.len() == 3 && item[0] == item[1] && item[1] == item[2])
}
#[inline]
fn is_pair(item: &[u8]) -> bool {
    item.len() == 2
}
fn has_pon_or_kan_of(hand: &[Vec<u8>], tile: u8) -> bool {
    hand.iter().any(|it| it[0] == tile && (it.len() == 4 || (it.len() == 3 && it[1] == tile)))
}
fn classify_hand_suits(hand: &[Vec<u8>]) -> (u8, u32) {
    let mut mask = 0u8;
    let mut honors = 0u32;
    for it in hand {
        let f = it[0];
        if f >= 27 {
            honors += 1;
        } else if f >= 18 {
            mask |= 1;
        } else if f >= 9 {
            mask |= 2;
        } else {
            mask |= 4;
        }
    }
    (mask, honors)
}
#[inline]
fn is_terminal(t: u8) -> bool {
    t < 27 && (t % 9 == 0 || t % 9 == 8)
}
#[inline]
fn is_terminal_or_honor34(t: u8) -> bool {
    t >= 27 || is_terminal(t)
}

fn indicator_to_dora(ind: u8) -> u8 {
    if ind < EAST {
        let base = ind / 9 * 9;
        base + (ind - base + 1) % 9
    } else if ind <= NORTH {
        EAST + (ind - EAST + 1) % 4
    } else {
        HAKU + (ind - HAKU + 1) % 3
    }
}

fn count_dora(tiles_34: &[u8; 34], indicators: &[u8]) -> u32 {
    let mut total = 0u32;
    for &ind in indicators {
        total += tiles_34[indicator_to_dora(ind) as usize] as u32;
    }
    total
}

// ---------------- fu ----------------
struct FuOut {
    details: Vec<u32>,
    total: u32,
}

fn calculate_fu(hand: &[Vec<u8>], win_tile: u8, win_group: &[u8], cfg: &Config, valued: &[u8],
                melds: &[ScoreMeld]) -> FuOut {
    if hand.len() == 7 {
        return FuOut { details: vec![25], total: 25 };
    }
    let mut details = Vec::new();
    let mut total = 0u32;
    let wg_is_chi = win_group.len() == 3 && win_group[0] != win_group[1];
    let mut pair_tile: Option<u8> = None;
    let mut pon_sets: Vec<&Vec<u8>> = Vec::new();
    let mut wg_chi_count_in_hand = 0;
    for grp in hand {
        if grp.len() == 2 {
            pair_tile = Some(grp[0]);
            continue;
        }
        if grp[0] == grp[1] {
            pon_sets.push(grp);
            continue;
        }
        if wg_is_chi && grp.len() == 3 && grp[..] == win_group[..] {
            wg_chi_count_in_hand += 1;
        }
    }
    let mut is_open_hand = false;
    let mut wg_open_chi_count = 0;
    // meld_state: bit0 opened, bit1 kan
    let mut meld_state = [0u8; 34];
    for m in melds {
        if m.opened {
            is_open_hand = true;
        }
        match m.mtype {
            MeldType::Chi => {
                if wg_is_chi && m.tiles[..] == win_group[..] {
                    wg_open_chi_count += 1;
                }
            }
            _ => {
                let mut st = 0u8;
                if m.opened {
                    st |= 1;
                }
                if matches!(m.mtype, MeldType::Kan | MeldType::Shouminkan) {
                    st |= 2;
                }
                meld_state[m.tiles[0] as usize] = st;
            }
        }
    }
    let wg_closed_chi = wg_is_chi && wg_chi_count_in_hand > wg_open_chi_count;
    if wg_closed_chi {
        let start_rank = win_group[0] % 9;
        let is_penchan = (start_rank == 0 && win_tile == win_group[2]) || (start_rank == 6 && win_tile == win_group[0]);
        if is_penchan {
            details.push(2);
            total += 2;
        }
        if win_tile == win_group[1] {
            details.push(2);
            total += 2;
        }
    }
    if let Some(pt) = pair_tile {
        let vc = valued.iter().filter(|&&v| v == pt).count();
        if vc == 1 {
            details.push(2);
            total += 2;
        } else if vc >= 2 {
            details.push(4);
            total += 4;
        }
    }
    if win_group.len() == 2 {
        details.push(2);
        total += 2;
    }
    for set in &pon_sets {
        let tile = set[0];
        let st = meld_state[tile as usize];
        let mut set_open = st & 1 != 0;
        let is_kan_set = set.len() == 4 || st & 2 != 0;
        if !cfg.is_tsumo && win_group.len() == set.len() && win_group[0] == tile && win_group[1] == tile {
            set_open = true;
        }
        let th = is_terminal_or_honor34(tile);
        let fu = match (th, is_kan_set, set_open) {
            (false, false, false) => 4,
            (false, false, true) => 2,
            (false, true, false) => 16,
            (false, true, true) => 8,
            (true, false, false) => 8,
            (true, false, true) => 4,
            (true, true, false) => 32,
            (true, true, true) => 16,
        };
        details.push(fu);
        total += fu;
    }
    if cfg.is_tsumo && total > 0 {
        details.push(2);
        total += 2;
    }
    if is_open_hand && total == 0 {
        details.push(2);
        total += 2;
    }
    let base = if is_open_hand || cfg.is_tsumo { 20 } else { 30 };
    details.push(base);
    total += base;
    FuOut { details, total: (total + 9) / 10 * 10 }
}

// ---------------- scores ----------------
fn calculate_scores(mut han: u32, fu: u32, cfg: &Config, is_yakuman: bool) -> (i64, i64) {
    // kazoe_limit default = KAZOE_LIMITED: 13+ han without a yakuman yaku counts as one yakuman
    if han >= 13 && !is_yakuman {
        han = 13;
    }
    let (rounded, double_rounded, four_rounded, six_rounded): (i64, i64, i64, i64);
    if han >= 5 {
        let r: i64 = if han >= 78 {
            48000 + (((han - 78) / 13) as i64) * 8000
        } else if han >= 65 { 40000 } else if han >= 52 { 32000 } else if han >= 39 { 24000 }
        else if han >= 26 { 16000 } else if han >= 13 { 8000 } else if han >= 11 { 6000 }
        else if han >= 8 { 4000 } else if han >= 6 { 3000 } else { 2000 };
        rounded = r;
        double_rounded = r * 2;
        four_rounded = double_rounded * 2;
        six_rounded = double_rounded * 3;
    } else {
        let base = (fu as i64) * (1i64 << (2 + han));
        let mut r = (base + 99) / 100 * 100;
        let mut d = (2 * base + 99) / 100 * 100;
        let mut f = (4 * base + 99) / 100 * 100;
        let mut s = (6 * base + 99) / 100 * 100;
        if r > 2000 {
            r = 2000;
            d = r * 2;
            f = d * 2;
            s = d * 3;
        }
        rounded = r;
        double_rounded = d;
        four_rounded = f;
        six_rounded = s;
    }
    if cfg.is_tsumo {
        let main = double_rounded;
        let additional = if cfg.is_dealer() { main } else { rounded };
        (main, additional)
    } else {
        (if cfg.is_dealer() { six_rounded } else { four_rounded }, 0)
    }
}

// ---------------- main entry ----------------
pub struct HandInput<'a> {
    /// every tile of the hand incl. melds and the win tile, as 34-indices
    pub tiles_34: [u8; 34],
    pub win_tile: u8,
    pub melds: &'a [ScoreMeld],
    pub dora_indicators: &'a [u8],
    pub ura_indicators: &'a [u8],
    pub aka_count: u32,
    pub cfg: Config,
}

struct Calculated {
    han: u32,
    fu: u32,
    yaku: Vec<Yaku>,
    fu_sum: u32,
    error: bool,
}

/// HandCalculator.estimate_hand_value for the engine's configuration.
/// `None` == a HandResponse with `error` set (not winning / no yaku).
pub fn estimate(inp: &HandInput) -> Option<ScoreResult> {
    let cfg = &inp.cfg;
    let opened_melds: Vec<Vec<u8>> = inp.melds.iter().filter(|m| m.opened).map(|m| m.tiles.clone()).collect();
    let is_open_hand = !opened_melds.is_empty();
    if inp.tiles_34[inp.win_tile as usize] == 0 {
        return None;
    }
    if cfg.is_riichi && !cfg.is_daburu_riichi && is_open_hand {
        return None;
    }
    if cfg.is_daburu_riichi && is_open_hand {
        return None;
    }
    if cfg.is_ippatsu && !cfg.is_riichi && !cfg.is_daburu_riichi {
        return None;
    }
    if cfg.is_chankan && cfg.is_tsumo {
        return None;
    }
    if cfg.is_rinshan && !cfg.is_tsumo {
        return None;
    }
    if cfg.is_haitei && !cfg.is_tsumo {
        return None;
    }
    if cfg.is_houtei && cfg.is_tsumo {
        return None;
    }
    if cfg.is_haitei && cfg.is_rinshan {
        return None;
    }
    if cfg.is_houtei && cfg.is_chankan {
        return None;
    }
    if cfg.is_tenhou && (!cfg.is_dealer() || !cfg.is_tsumo || !inp.melds.is_empty()) {
        return None;
    }
    if cfg.is_chiihou && (cfg.is_dealer() || !cfg.is_tsumo || !inp.melds.is_empty()) {
        return None;
    }
    let hand_options = divide_hand(&inp.tiles_34, inp.melds);
    let dora_n = count_dora(&inp.tiles_34, inp.dora_indicators);
    let aka_n = inp.aka_count;
    let ura_n = if cfg.is_riichi || cfg.is_daburu_riichi { count_dora(&inp.tiles_34, inp.ura_indicators) } else { 0 };
    let win = inp.win_tile;
    let mut calculated: Vec<Calculated> = Vec::new();

    for blocks in &hand_options {
        let hand: Vec<Vec<u8>> = blocks.iter().map(|&b| block_tiles(b)).collect();
        let is_chiitoitsu = hand.len() == 7;
        let valued = [HAKU, HATSU, CHUN, cfg.player_wind, cfg.round_wind];
        let mut chi_sets = 0;
        let mut pon_sets = 0;
        let mut kan_sets = 0;
        for x in &hand {
            match x.len() {
                4 => kan_sets += 1,
                3 => {
                    if x[0] == x[1] { pon_sets += 1 } else { chi_sets += 1 }
                }
                _ => {}
            }
        }
        let is_tanyao_hand = hand.iter().flatten().all(|&t| !is_terminal_or_honor34(t));
        let (suit_mask, honor_count) = classify_hand_suits(&hand);
        let has_honors = honor_count > 0;
        // _find_win_groups
        let mut consumed = vec![false; opened_melds.len()];
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut win_groups: Vec<Vec<u8>> = Vec::new();
        for x in &hand {
            let mut is_open = false;
            for (i, m) in opened_melds.iter().enumerate() {
                if !consumed[i] && x == m {
                    consumed[i] = true;
                    is_open = true;
                    break;
                }
            }
            if is_open {
                continue;
            }
            if x.contains(&win) && !seen.contains(x) {
                seen.push(x.clone());
                win_groups.push(x.clone());
            }
        }
        for wg in &win_groups {
            let fu = calculate_fu(&hand, win, wg, cfg, &valued, inp.melds);
            let is_pinfu = fu.details.len() == 1 && !is_chiitoitsu && !is_open_hand;
            let mut y: Vec<Yaku> = Vec::new();
            if cfg.is_tsumo && !is_open_hand { y.push(Yaku::Tsumo); }
            if is_pinfu { y.push(Yaku::Pinfu); }
            if is_chiitoitsu { y.push(Yaku::Chiitoitsu); }
            if is_tanyao_hand { y.push(Yaku::Tanyao); }     // has_open_tanyao
            if cfg.is_riichi && !cfg.is_daburu_riichi { y.push(Yaku::Riichi); }
            if cfg.is_daburu_riichi { y.push(Yaku::DaburuRiichi); }
            if cfg.is_ippatsu { y.push(Yaku::Ippatsu); }
            if cfg.is_rinshan { y.push(Yaku::Rinshan); }
            if cfg.is_chankan { y.push(Yaku::Chankan); }
            if cfg.is_haitei { y.push(Yaku::Haitei); }
            if cfg.is_houtei { y.push(Yaku::Houtei); }
            if cfg.is_tenhou { y.push(Yaku::Tenhou); }
            if cfg.is_chiihou { y.push(Yaku::Chiihou); }
            if honor_count == 0 && matches!(suit_mask, 1 | 2 | 4) {
                y.push(Yaku::Chinitsu);
            } else if matches!(suit_mask, 1 | 2 | 4) && honor_count > 0 {
                y.push(Yaku::Honitsu);
            }
            if chi_sets == 0 {
                if has_honors && hand.iter().flatten().all(|&t| t >= 27) { y.push(Yaku::Tsuisou); }
                if !is_tanyao_hand {
                    if hand.iter().flatten().all(|&t| is_terminal_or_honor34(t)) { y.push(Yaku::Honroto); }
                    if !has_honors && hand.iter().flatten().all(|&t| is_terminal(t)) { y.push(Yaku::Chinroto); }
                }
            }
            // ryuisou: 2s 3s 4s 6s 8s hatsu
            if hand.iter().flatten().all(|&t| matches!(t, 19 | 20 | 21 | 23 | 25 | 32)) { y.push(Yaku::Ryuisou); }
            if chi_sets > 0 {
                if !is_tanyao_hand {
                    // chantai
                    let (mut hs, mut ts, mut nchi) = (0, 0, 0);
                    for it in &hand {
                        if is_chi(it) { nchi += 1; }
                        if is_terminal(it[0]) || is_terminal(*it.last().unwrap()) { ts += 1; }
                        else if it[0] >= 27 { hs += 1; }
                    }
                    if nchi > 0 && ts + hs == 5 && ts != 0 && hs != 0 { y.push(Yaku::Chantai); }
                    if nchi > 0 && ts == 5 { y.push(Yaku::Junchan); }
                    // ittsu
                    let mut masks = [0u8; 3];
                    for it in &hand {
                        if it[0] + 1 != it[1] { continue; }
                        let bit = match it[0] % 9 { 0 => 1, 3 => 2, 6 => 4, _ => continue };
                        masks[(it[0] / 9) as usize] |= bit;
                    }
                    if masks.iter().any(|&m| m == 7) { y.push(Yaku::Ittsu); }
                }
                if !is_open_hand {
                    let mut counts = [0u8; 34];
                    for it in &hand {
                        if is_chi(it) { counts[it[0] as usize] += 1; }
                    }
                    let pairs: u32 = counts.iter().map(|&c| (c / 2) as u32).sum();
                    if pairs >= 2 { y.push(Yaku::Ryanpeiko); }
                    else if counts.iter().any(|&c| c >= 2) { y.push(Yaku::Iipeiko); }
                }
                if suit_mask == 7 {
                    let mut m = [0u16; 3];
                    for it in &hand {
                        if it[0] + 1 != it[1] { continue; }
                        m[(it[0] / 9) as usize] |= 1 << (it[0] % 9);
                    }
                    if m[0] & m[1] & m[2] != 0 { y.push(Yaku::Sanshoku); }
                }
            }
            if pon_sets > 0 || kan_sets > 0 {
                if hand.iter().filter(|it| is_pon_or_kan(it)).count() == 4 { y.push(Yaku::Toitoi); }
                // sanankou
                {
                    let mut has_chi_with_win = false;
                    let mut closed_pon = 0i32;
                    for it in &hand {
                        if is_pon_or_kan(it) {
                            if !opened_melds.contains(it) { closed_pon += 1; }
                        } else if is_chi(it) && it.contains(&win) && !opened_melds.contains(it) {
                            has_chi_with_win = true;
                        }
                    }
                    if !cfg.is_tsumo && !has_chi_with_win {
                        for it in &hand {
                            if is_pon_or_kan(it) && it[0] == win && !opened_melds.contains(it) {
                                closed_pon -= 1;
                                break;
                            }
                        }
                    }
                    if closed_pon == 3 { y.push(Yaku::Sanankou); }
                }
                if suit_mask == 7 {
                    let mut m = [0u16; 3];
                    for it in &hand {
                        if it[0] >= 27 || !is_pon_or_kan(it) { continue; }
                        m[(it[0] / 9) as usize] |= 1 << (it[0] % 9);
                    }
                    if m[0] & m[1] & m[2] != 0 { y.push(Yaku::SanshokuDouko); }
                }
                if has_honors {
                    let dragons = hand.iter().filter(|it| it[0] >= HAKU && (is_pair(it) || is_pon_or_kan(it))).count();
                    if dragons == 3 { y.push(Yaku::Shosangen); }
                    if has_pon_or_kan_of(&hand, HAKU) { y.push(Yaku::Haku); }
                    if has_pon_or_kan_of(&hand, HATSU) { y.push(Yaku::Hatsu); }
                    if has_pon_or_kan_of(&hand, CHUN) { y.push(Yaku::Chun); }
                    for w in [EAST, SOUTH, WEST, NORTH] {
                        if cfg.player_wind == w && has_pon_or_kan_of(&hand, w) { y.push(Yaku::SeatWind(w)); }
                    }
                    for w in [EAST, SOUTH, WEST, NORTH] {
                        if cfg.round_wind == w && has_pon_or_kan_of(&hand, w) { y.push(Yaku::RoundWind(w)); }
                    }
                    let dragon_pons = hand.iter().filter(|it| it[0] >= HAKU && is_pon_or_kan(it)).count();
                    if dragon_pons == 3 { y.push(Yaku::Daisangen); }
                    let mut wind_sets = 0;
                    let mut wind_pair = 0;
                    for it in &hand {
                        if it[0] < EAST || it[0] > NORTH { continue; }
                        if is_pair(it) { wind_pair += 1; } else if is_pon_or_kan(it) { wind_sets += 1; }
                    }
                    if wind_sets == 3 && wind_pair == 1 { y.push(Yaku::Shosuushi); }
                    if wind_sets == 4 { y.push(Yaku::Daisuushi); }
                }
                if inp.melds.is_empty() && !is_tanyao_hand && chuuren(&hand) {
                    let c = inp.tiles_34[win as usize];
                    if c == 2 || c == 4 { y.push(Yaku::DaburuChuuren); } else { y.push(Yaku::Chuuren); }
                }
                if !is_open_hand {
                    let closed: Vec<&Vec<u8>> = hand.iter()
                        .filter(|it| !(!cfg.is_tsumo && it.contains(&win) && is_pon_or_kan(it))).collect();
                    if closed.iter().filter(|it| is_pon_or_kan(it)).count() == 4 {
                        if inp.tiles_34[win as usize] == 2 { y.push(Yaku::SuuankouTanki); } else { y.push(Yaku::Suuankou); }
                    }
                }
                let kans = inp.melds.iter().filter(|m| matches!(m.mtype, MeldType::Kan | MeldType::Shouminkan)).count();
                if kans == 3 { y.push(Yaku::Sankantsu); }
                if kans == 4 { y.push(Yaku::Suukantsu); }
            }
            let yakuman: Vec<Yaku> = y.iter().copied().filter(|k| k.is_yakuman()).collect();
            let has_yakuman = !yakuman.is_empty();
            if has_yakuman {
                y = yakuman;
            }
            let mut han: u32 = y.iter().map(|k| k.han(is_open_hand)).sum();
            let error = han == 0;
            if !has_yakuman && !error {
                if dora_n > 0 { y.push(Yaku::Dora(dora_n)); han += dora_n; }
                if aka_n > 0 { y.push(Yaku::AkaDora(aka_n)); han += aka_n; }
                if ura_n > 0 { y.push(Yaku::UraDora(ura_n)); han += ura_n; }
            }
            let fu_sum: u32 = fu.details.iter().sum();
            calculated.push(Calculated { han, fu: fu.total, yaku: y, fu_sum, error });
        }
    }
    // kokushi
    if !is_open_hand {
        let t = &inp.tiles_34;
        let prod: u32 = [0usize, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33].iter().map(|&i| t[i] as u32).product();
        if prod == 2 {
            let mut y = vec![if t[win as usize] == 2 { Yaku::DaburuKokushi } else { Yaku::Kokushi }];
            if cfg.is_tenhou { y.push(Yaku::Tenhou); }
            if cfg.is_chiihou { y.push(Yaku::Chiihou); }
            let han: u32 = y.iter().map(|k| k.han(false)).sum();
            calculated.push(Calculated { han, fu: 0, yaku: y, fu_sum: 0, error: false });
        }
    }
    if calculated.is_empty() {
        return None;
    }
    // stable sort by (han, fu) desc, keep ties, stable sort by fu_sum desc, first wins
    let best_key = calculated.iter().map(|c| (c.han, c.fu)).max().unwrap();
    let mut best_idx: Option<usize> = None;
    let mut best_sum = 0u32;
    for (i, c) in calculated.iter().enumerate() {
        if (c.han, c.fu) != best_key {
            continue;
        }
        if best_idx.is_none() || c.fu_sum > best_sum {
            best_idx = Some(i);
            best_sum = c.fu_sum;
        }
    }
    let c = &calculated[best_idx.unwrap()];
    if c.error {
        return None;
    }
    let is_yakuman = c.yaku.iter().any(|k| k.is_yakuman());
    let (main, additional) = calculate_scores(c.han, c.fu, cfg, is_yakuman);
    let mut yaku = c.yaku.clone();
    yaku.sort_by_key(|y| y.id());          // HandResponse orders by yaku_id
    Some(ScoreResult { han: c.han, fu: c.fu, yaku, main, additional })
}

fn chuuren(hand: &[Vec<u8>]) -> bool {
    let (mask, honors) = classify_hand_suits(hand);
    if honors > 0 || !matches!(mask, 1 | 2 | 4) {
        return false;
    }
    let mut counts = [0i32; 9];
    for &t in hand.iter().flatten() {
        counts[(t % 9) as usize] += 1;
    }
    if counts[0] < 3 || counts[8] < 3 {
        return false;
    }
    counts[0] -= 2;
    counts[8] -= 2;
    let mut two = false;
    for &c in &counts {
        match c {
            1 => {}
            2 => {
                if two {
                    return false;
                }
                two = true;
            }
            _ => return false,
        }
    }
    two
}
