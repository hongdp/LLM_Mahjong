//! PyMahjongTable port. Phase P1: deal (reset) parity. Later phases add the
//! turn/interrupt machinery, settlement and scoring.
use crate::pyrandom::Mt19937;
use crate::tiles::*;

pub const REWARD_SCALE: f64 = 0.001;
pub const RANK_BONUS: [f64; 4] = [2.0, 0.5, -0.5, -2.0];
pub const ILLEGAL_PENALTY: f64 = -5.0;
pub const FORMAT_PENALTY: f64 = -10.0;
pub const MAX_MELDS: usize = 4;
pub const MAX_KANS: usize = 4;
pub const RIICHI_MIN_WALL: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeldKind {
    Chi,
    Pon,
    Kan,        // daiminkan (open)
    Ankan,
    Shouminkan,
}

#[derive(Clone, Debug)]
pub struct Meld {
    pub kind: MeldKind,
    pub tiles: Vec<Tile>, // normalized, chi sorted by sort_key
    pub opened: bool,
    pub from: Option<usize>,
    pub red: u8,
}

#[derive(Clone, Debug)]
pub struct RiverEvent {
    pub tile: Tile,       // spelled: red code kept
    pub tsumogiri: bool,
    pub riichi: bool,
    pub called: bool,
    pub idx: usize,
}

#[derive(Clone)]
pub struct Table {
    pub rng: Mt19937,
    pub randomize_round: bool,
    pub dealer: usize,
    pub round_wind_idx: usize,
    pub round_number: usize,
    pub turn: usize,
    pub points: [i64; 4],
    pub start_points: [i64; 4],
    pub kyotaku: i64,
    pub start_kyotaku: i64,
    pub honba: i64,
    pub discards: [Vec<(Tile, bool)>; 4],   // (spelled tile, riichi mark)
    pub furiten_river: [Vec<Tile>; 4],      // normalized
    pub melds: [Vec<Meld>; 4],
    pub riichi: [bool; 4],
    pub riichi_pending: Option<usize>,
    pub river_events: [Vec<RiverEvent>; 4],
    pub riichi_turn: [Option<usize>; 4],
    pub ippatsu: [bool; 4],
    pub daburu: [bool; 4],
    pub temp_furiten: [bool; 4],
    pub perm_furiten: [bool; 4],
    pub rinshan: [bool; 4],
    pub discard_count: [usize; 4],
    pub any_call: bool,
    pub kan_count: usize,
    pub kuikae: Option<(usize, Vec<Tile>)>,
    pub pending_kan: Option<PendingKan>,
    pub pao: [Option<usize>; 4],
    pub pending_dora_reveal: usize,
    pub kan_players: [bool; 4],
    pub pending_abort: Option<String>,
    pub ron_chance: [bool; 4],
    pub last_discard: Option<Tile>,        // normalized
    pub last_discard_red: bool,
    pub last_discarder: Option<usize>,
    pub last_drawn: [Option<Tile>; 4],     // normalized
    pub last_drawn_red: [bool; 4],
    pub finished: bool,
    pub final_rewards: Option<[f64; 4]>,
    pub result_summary: String,
    pub wall: Vec<Tile>,        // spelled (red codes), pop() from the END like Python
    pub dead_wall: Vec<Tile>,   // 14 slots; [0:4] rinshan raw, [4:14] indicators normalized
    pub rinshan_idx: usize,
    pub dora_indicators: Vec<Tile>,
    pub ura_indicators: Vec<Tile>,
    pub hands: [Vec<Tile>; 4],  // normalized, sorted by sort_key
    pub red: [[u8; 3]; 4],      // per seat: m, p, s red fives held
}

#[derive(Clone, Debug)]
pub struct PendingKan {
    pub player: usize,
    pub tile: Tile,
    pub ankan: bool,
    pub red: bool,
}

/// CPython `round()` (half to even) on x/100 then *100, as in
/// `int(round(x / 100.0)) * 100`.
fn round_hundred(x: f64) -> i64 {
    let v = x / 100.0;
    let r = v.round_ties_even();
    (r as i64) * 100
}

impl Table {
    /// Equivalent of `random.seed(seed); PyMahjongTable(randomize_round=...)`.
    pub fn new_seeded(seed: u64, randomize_round: bool) -> Table {
        let mut rng = Mt19937::new();
        rng.seed_u64(seed);
        let mut t = Table::blank(rng, randomize_round);
        t.reset();
        t
    }

    /// Table whose RNG state is supplied by the caller (a shared stream).
    pub fn with_rng(rng: Mt19937, randomize_round: bool) -> Table {
        let mut t = Table::blank(rng, randomize_round);
        t.reset();
        t
    }

    fn blank(rng: Mt19937, randomize_round: bool) -> Table {
        Table {
            rng,
            randomize_round,
            dealer: 0,
            round_wind_idx: 0,
            round_number: 1,
            turn: 0,
            points: [25000; 4],
            start_points: [25000; 4],
            kyotaku: 0,
            start_kyotaku: 0,
            honba: 0,
            discards: Default::default(),
            furiten_river: Default::default(),
            melds: Default::default(),
            riichi: [false; 4],
            riichi_pending: None,
            river_events: Default::default(),
            riichi_turn: [None; 4],
            ippatsu: [false; 4],
            daburu: [false; 4],
            temp_furiten: [false; 4],
            perm_furiten: [false; 4],
            rinshan: [false; 4],
            discard_count: [0; 4],
            any_call: false,
            kan_count: 0,
            kuikae: None,
            pending_kan: None,
            pao: [None; 4],
            pending_dora_reveal: 0,
            kan_players: [false; 4],
            pending_abort: None,
            ron_chance: [false; 4],
            last_discard: None,
            last_discard_red: false,
            last_discarder: None,
            last_drawn: [None; 4],
            last_drawn_red: [false; 4],
            finished: false,
            final_rewards: None,
            result_summary: String::new(),
            wall: Vec::with_capacity(136),
            dead_wall: Vec::with_capacity(14),
            rinshan_idx: 0,
            dora_indicators: Vec::new(),
            ura_indicators: Vec::new(),
            hands: Default::default(),
            red: [[0; 3]; 4],
        }
    }

    /// Mirrors PyMahjongTable.reset() including its RNG consumption order.
    pub fn reset(&mut self) {
        if self.randomize_round {
            self.dealer = self.rng.randrange(4) as usize;
            let r = self.rng.random();
            self.round_wind_idx = if r < 0.45 { 0 } else if r < 0.9 { 1 } else { 2 };
        } else {
            self.dealer = 0;
            self.round_wind_idx = 0;
        }
        self.round_number = self.dealer + 1;
        self.turn = self.dealer;

        let k = self.round_wind_idx * 4 + self.dealer;
        if self.randomize_round && k > 0 {
            let spread = self.rng.uniform(0.5, 1.5) * 4500.0 * (k as f64).sqrt();
            let mut z = [0.0f64; 4];
            for zi in z.iter_mut() {
                *zi = self.rng.gauss(0.0, 1.0).clamp(-2.2, 2.2);
            }
            let zm = z.iter().sum::<f64>() / 4.0;
            let mut pts = [0i64; 4];
            for i in 0..4 {
                pts[i] = 25000 + round_hundred(spread * (z[i] - zm));
            }
            let sum: i64 = pts.iter().sum();
            let imax = argmax(&pts);
            pts[imax] += 100000 - sum;
            loop {
                let imin = argmin(&pts);
                if pts[imin] >= 0 {
                    break;
                }
                let imax = argmax(&pts);
                pts[imax] += pts[imin];
                pts[imin] = 0;
            }
            self.points = pts;
            let c = self.rng.choices_index(&[70.0, 20.0, 8.0, 2.0]);
            self.kyotaku = 1000 * c as i64;
        } else {
            self.points = [25000; 4];
            self.kyotaku = 0;
        }
        self.start_points = self.points;
        self.start_kyotaku = self.kyotaku;

        for i in 0..4 {
            self.discards[i].clear();
            self.furiten_river[i].clear();
            self.melds[i].clear();
            self.river_events[i].clear();
            self.hands[i].clear();
        }
        self.riichi = [false; 4];
        self.riichi_pending = None;
        self.riichi_turn = [None; 4];
        self.ippatsu = [false; 4];
        self.daburu = [false; 4];
        self.temp_furiten = [false; 4];
        self.perm_furiten = [false; 4];
        self.rinshan = [false; 4];
        self.discard_count = [0; 4];
        self.any_call = false;
        self.kan_count = 0;
        self.kuikae = None;
        self.pending_kan = None;
        self.pao = [None; 4];
        self.pending_dora_reveal = 0;
        self.kan_players = [false; 4];
        self.pending_abort = None;
        self.ron_chance = [false; 4];
        self.last_discard = None;
        self.last_discard_red = false;
        self.last_discarder = None;
        self.last_drawn = [None; 4];
        self.last_drawn_red = [false; 4];
        self.finished = false;
        self.final_rewards = None;
        self.result_summary.clear();

        // all_tiles = [1..9 m, 1..9 p, 1..9 s, 1..7 z] * 4, then the FIRST 5m/5p/5s
        // (index of '5m' in the flat list = position 4 / 13 / 22) become red.
        let mut wall: Vec<Tile> = Vec::with_capacity(136);
        for _ in 0..4 {
            for t in 0..34u8 {
                wall.push(t);
            }
        }
        wall[4] = RED_M;
        wall[13] = RED_P;
        wall[22] = RED_S;
        self.rng.shuffle(&mut wall);
        self.wall = wall;
        self.red = [[0; 3]; 4];
        // dead wall: 14 pops from the end; indicator slots (>= 4) normalized
        let mut dead = Vec::with_capacity(14);
        for _ in 0..14 {
            dead.push(self.wall.pop().unwrap());
        }
        for (i, t) in dead.iter_mut().enumerate() {
            if i >= 4 {
                *t = norm(*t);
            }
        }
        self.dead_wall = dead;
        self.rinshan_idx = 0;
        self.dora_indicators = vec![self.dead_wall[4]];
        self.ura_indicators = vec![self.dead_wall[9]];

        for pid in 0..4 {
            for _ in 0..13 {
                let raw = self.wall.pop().unwrap();
                self.give(pid, raw);
            }
            sort_hand(&mut self.hands[pid]);
        }
        let raw = self.wall.pop().unwrap();
        let first = self.give(self.dealer, raw);
        sort_hand(&mut self.hands[self.dealer]);
        self.last_drawn[self.dealer] = Some(first);
    }

    /// Move a wall tile into a hand; returns the normalized tile.
    pub fn give(&mut self, pid: usize, raw: Tile) -> Tile {
        if is_red(raw) {
            self.red[pid][(norm(raw) / 9) as usize] += 1;
        }
        let t = norm(raw);
        self.hands[pid].push(t);
        self.last_drawn_red[pid] = is_red(raw);
        t
    }
}

fn argmax(v: &[i64; 4]) -> usize {
    let mut b = 0;
    for i in 1..4 {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

fn argmin(v: &[i64; 4]) -> usize {
    let mut b = 0;
    for i in 1..4 {
        if v[i] < v[b] {
            b = i;
        }
    }
    b
}

// ======================================================================
// Phase P3: game logic (port of PyMahjongTable turn/interrupt machinery)
// ======================================================================
use crate::score::{self, Config, ScoreMeld, ScoreResult, MeldType};
use crate::shanten;

#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub discarded: bool,
    pub chankan: Option<Tile>,
    pub interrupt: bool,
    pub winners: Vec<usize>,
    pub abort: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ParsedAction {
    pub kind: String,
    pub tile: Option<String>,
    pub with: Option<String>,
}

/// ACTION_RE: `<action type="X"( tile="Y")?( with="Z")? />` — attribute order fixed.
pub fn parse_action(xml: &str) -> Option<ParsedAction> {
    let i = xml.find("<action")?;
    let rest = &xml[i + 7..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix("type=\"")?;
    let e = rest.find('"')?;
    let kind = rest[..e].to_string();
    let mut rest = &rest[e + 1..];
    let mut tile = None;
    let mut with = None;
    let r2 = rest.trim_start();
    if let Some(r) = r2.strip_prefix("tile=\"") {
        let e = r.find('"')?;
        tile = Some(r[..e].to_string());
        rest = &r[e + 1..];
    }
    let r3 = rest.trim_start();
    if let Some(r) = r3.strip_prefix("with=\"") {
        let e = r.find('"')?;
        with = Some(r[..e].to_string());
        rest = &r[e + 1..];
    }
    let tail = rest.trim_start();
    if !tail.starts_with("/>") {
        return None;
    }
    Some(ParsedAction { kind, tile, with })
}

fn fmt_points(p: &[i64; 4]) -> String {
    format!("[{}, {}, {}, {}]", p[0], p[1], p[2], p[3])
}

fn fmt_list(v: &[usize]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn is_orphan(t: Tile) -> bool {
    is_terminal_or_honor(t)
}

fn kyuushu_kinds(hand: &[Tile]) -> usize {
    let mut seen = [false; 34];
    let mut n = 0;
    for &t in hand {
        if is_orphan(t) && !seen[norm(t) as usize] {
            seen[norm(t) as usize] = true;
            n += 1;
        }
    }
    n
}

fn spelled(tile: Tile, red: bool) -> String {
    if red {
        format!("0{}", to_string(tile).chars().nth(1).unwrap())
    } else {
        to_string(tile)
    }
}

impl Table {
    // ---- bookkeeping helpers ----
    fn plain_copies(&self, pid: usize, tile: Tile) -> i32 {
        let mut n = self.hands[pid].iter().filter(|&&t| t == tile).count() as i32;
        if rank(tile) == 5 && !is_honor(tile) {
            n -= self.red[pid][suit(tile) as usize] as i32;
        }
        n
    }

    fn take_from_hand(&mut self, pid: usize, tile: Tile, n: usize, prefer_red: bool) -> u8 {
        for _ in 0..n {
            if let Some(pos) = self.hands[pid].iter().position(|&t| t == tile) {
                self.hands[pid].remove(pos);
            }
        }
        let mut used_red = 0i32;
        if rank(tile) == 5 && !is_honor(tile) {
            let s = suit(tile) as usize;
            let plain = self.hands[pid].iter().filter(|&&t| t == tile).count() as i32 + n as i32 - self.red[pid][s] as i32;
            used_red = if prefer_red { n as i32 } else { (n as i32 - plain).max(0) };
            used_red = used_red.min(self.red[pid][s] as i32);
            self.red[pid][s] -= used_red as u8;
        }
        used_red as u8
    }

    pub fn shanten_of(&self, tiles: &[Tile], num_melds: usize) -> i32 {
        shanten::shanten_hand(tiles, num_melds)
    }

    pub fn waits(&self, pid: usize) -> Vec<Tile> {
        shanten::waits_of(&self.hands[pid], self.melds[pid].len())
    }

    fn is_furiten(&self, pid: usize) -> bool {
        if self.perm_furiten[pid] || self.temp_furiten[pid] {
            return true;
        }
        let waits = self.waits(pid);
        waits.iter().any(|w| self.furiten_river[pid].contains(w))
    }

    fn is_closed(&self, pid: usize) -> bool {
        self.melds[pid].iter().all(|m| m.kind == MeldKind::Ankan)
    }

    fn situational_config(&self, pid: usize, is_tsumo: bool, chankan: bool) -> Config {
        let virgin = !self.any_call;
        let no_discard_yet = self.discard_count[pid] == 0;
        let is_dealer = pid == self.dealer;
        let total_discards: usize = self.discard_count.iter().sum();
        let tenhou = is_tsumo && is_dealer && virgin && total_discards == 0;
        let chiihou = is_tsumo && !is_dealer && virgin && no_discard_yet;
        let rinshan = is_tsumo && self.rinshan[pid];
        let last_tile = self.wall.is_empty();
        Config {
            is_tsumo,
            is_riichi: self.riichi[pid],
            is_daburu_riichi: self.daburu[pid],
            is_ippatsu: self.ippatsu[pid],
            is_rinshan: rinshan,
            is_chankan: chankan,
            is_haitei: is_tsumo && last_tile && !rinshan,
            is_houtei: !is_tsumo && last_tile && !chankan,
            is_tenhou: tenhou,
            is_chiihou: chiihou,
            player_wind: 27 + ((pid + 4 - self.dealer) % 4) as u8,
            round_wind: 27 + self.round_wind_idx.min(3) as u8,
        }
    }

    /// PyMahjongTable._win_result
    pub fn win_result(&self, pid: usize, win_tile: Tile, is_tsumo: bool, chankan: bool) -> Option<ScoreResult> {
        let hand = &self.hands[pid];
        let n_melds = self.melds[pid].len();
        let mut concealed: Vec<Tile>;
        if is_tsumo {
            if !hand.contains(&win_tile) {
                return None;
            }
            concealed = hand.clone();
        } else {
            if !self.waits(pid).contains(&win_tile) {
                return None;
            }
            concealed = hand.clone();
            concealed.push(win_tile);
        }
        if self.shanten_of(&concealed, n_melds) != shanten::AGARI {
            return None;
        }
        let mut rest = concealed.clone();
        if let Some(pos) = rest.iter().position(|&t| t == win_tile) {
            rest.remove(pos);
        }
        let mut reds: [i32; 3] = [self.red[pid][0] as i32, self.red[pid][1] as i32, self.red[pid][2] as i32];
        let win_red = if is_tsumo {
            self.last_drawn_red[pid] && Some(win_tile) == self.last_drawn[pid]
        } else if chankan && self.pending_kan.is_some() {
            self.pending_kan.as_ref().unwrap().red
        } else {
            self.last_discard_red
        };
        if win_red && is_tsumo {
            reds[suit(win_tile) as usize] -= 1;
        }
        // aka count as _alloc_136 would assign red ids: fives in `rest` take reds while available
        let mut aka: u32 = 0;
        for s in 0..3usize {
            let fives = rest.iter().filter(|&&t| rank(t) == 5 && suit(t) as usize == s && !is_honor(t)).count() as i32;
            aka += fives.min(reds[s].max(0)) as u32;
        }
        if win_red {
            aka += 1;
        }
        let mut counts = [0u8; 34];
        for &t in &concealed {
            counts[t as usize] += 1;
        }
        let mut smelds = Vec::with_capacity(n_melds);
        for m in &self.melds[pid] {
            for &t in &m.tiles {
                counts[t as usize] += 1;
            }
            // _alloc_136 hands the meld's red count to its fives (a chi 345 with red=1 -> the 5)
            if m.red > 0 && !is_honor(m.tiles[0]) {
                let fives = m.tiles.iter().filter(|&&t| rank(t) == 5).count();
                aka += (m.red as usize).min(fives) as u32;
            }
            let (mtype, opened) = match m.kind {
                MeldKind::Chi => (MeldType::Chi, true),
                MeldKind::Pon => (MeldType::Pon, true),
                MeldKind::Kan => (MeldType::Kan, true),
                MeldKind::Ankan => (MeldType::Kan, false),
                MeldKind::Shouminkan => (MeldType::Shouminkan, true),
            };
            smelds.push(ScoreMeld { mtype, tiles: m.tiles.clone(), opened });
        }
        let mut indicators: Vec<u8> = self.dora_indicators.iter().map(|&t| norm(t)).collect();
        if is_tsumo && self.pending_dora_reveal > 0 {
            let k = indicators.len();
            let end = (k + self.pending_dora_reveal).min(5);
            for i in k..end {
                indicators.push(norm(self.dead_wall[4 + i]));
            }
        }
        if self.riichi[pid] {
            let n = indicators.len().min(self.ura_indicators.len());
            for i in 0..n {
                indicators.push(norm(self.ura_indicators[i]));
            }
        }
        let cfg = self.situational_config(pid, is_tsumo, chankan);
        let inp = score::HandInput {
            tiles_34: counts,
            win_tile,
            melds: &smelds,
            dora_indicators: &indicators,
            ura_indicators: &[],
            aka_count: aka,
            cfg,
        };
        score::estimate(&inp)
    }

    fn can_ron(&self, pid: usize, tile: Tile, chankan: bool) -> bool {
        if self.is_furiten(pid) {
            return false;
        }
        self.win_result(pid, tile, false, chankan).is_some()
    }

    fn kan_allowed(&self, pid: usize, new_meld: bool) -> bool {
        self.kan_count < MAX_KANS && (!new_meld || self.melds[pid].len() < MAX_MELDS) && !self.wall.is_empty()
    }

    fn can_ankan(&self, pid: usize, tile: Tile) -> bool {
        if !self.kan_allowed(pid, true) {
            return false;
        }
        let Some(drawn) = self.last_drawn[pid] else { return false };
        if self.hands[pid].iter().filter(|&&t| t == tile).count() != 4 {
            return false;
        }
        if !self.riichi[pid] {
            return true;
        }
        if drawn != tile {
            return false;
        }
        let mut before = self.hands[pid].clone();
        if let Some(pos) = before.iter().position(|&t| t == drawn) {
            before.remove(pos);
        }
        let n_melds = self.melds[pid].len();
        let mut waits_before = shanten::waits_of(&before, n_melds);
        if waits_before.is_empty() {
            return false;
        }
        for &w in &waits_before {
            let mut hw = before.clone();
            hw.push(w);
            if !shanten::tile_only_as_triplet(&hw, tile, 4 - n_melds as i32) {
                return false;
            }
        }
        let after: Vec<Tile> = self.hands[pid].iter().copied().filter(|&t| t != tile).collect();
        let mut waits_after = shanten::waits_of(&after, n_melds + 1);
        waits_before.sort();
        waits_after.sort();
        waits_before == waits_after
    }

    fn can_shouminkan(&self, pid: usize, tile: Tile) -> bool {
        if !self.kan_allowed(pid, false) {
            return false;
        }
        if self.riichi[pid] || self.last_drawn[pid].is_none() {
            return false;
        }
        if !self.hands[pid].contains(&tile) {
            return false;
        }
        self.melds[pid].iter().any(|m| m.kind == MeldKind::Pon && m.tiles[0] == tile)
    }

    fn kuikae_tiles(&self, called: Tile, used: &[Tile]) -> Vec<Tile> {
        let mut forbidden = vec![called];
        if used.len() == 2 && suit(used[0]) == suit(used[1]) && !is_honor(used[0]) {
            let (a, b) = (rank(used[0]).min(rank(used[1])), rank(used[0]).max(rank(used[1])));
            let base = norm(used[0]) / 9 * 9;
            if b == a + 1 {
                for end in [a as i32 - 1, b as i32 + 1] {
                    if (1..=9).contains(&end) {
                        let t = base + (end - 1) as u8;
                        if t != called {
                            forbidden.push(t);
                        }
                    }
                }
            }
        }
        forbidden
    }

    fn forbidden_discards(&self, pid: usize) -> Vec<Tile> {
        match &self.kuikae {
            Some((p, v)) if *p == pid => v.clone(),
            _ => Vec::new(),
        }
    }

    fn discardable(&self, pid: usize, banned: &[Tile]) -> Vec<String> {
        let mut uniq: Vec<Tile> = Vec::new();
        for &t in &self.hands[pid] {
            if !uniq.contains(&t) {
                uniq.push(t);
            }
        }
        uniq.sort_by_key(|&t| sort_key(t));
        let mut out = Vec::new();
        for t in uniq {
            if banned.contains(&t) {
                continue;
            }
            if rank(t) == 5 && !is_honor(t) && self.red[pid][suit(t) as usize] > 0 {
                if self.plain_copies(pid, t) > 0 {
                    out.push(to_string(t));
                }
                out.push(spelled(t, true));
            } else {
                out.push(to_string(t));
            }
        }
        out
    }

    pub fn get_legal_actions(&self, pid: usize) -> Vec<String> {
        let hand = &self.hands[pid];
        let drawn = self.last_drawn[pid];
        let banned = self.forbidden_discards(pid);
        if self.riichi[pid] {
            let mut actions = Vec::new();
            if let Some(d) = drawn {
                if self.win_result(pid, d, true, false).is_some() {
                    actions.push("<action type=\"tsumo\" />".to_string());
                }
                if self.can_ankan(pid, d) {
                    actions.push(format!("<action type=\"kan\" tile=\"{}\" />", to_string(d)));
                }
                actions.push(format!("<action type=\"discard\" tile=\"{}\" />", spelled(d, self.last_drawn_red[pid])));
            }
            if actions.is_empty() {
                return self.discardable(pid, &banned).into_iter()
                    .map(|t| format!("<action type=\"discard\" tile=\"{t}\" />")).collect();
            }
            return actions;
        }
        let uniq = self.discardable(pid, &banned);
        let mut actions: Vec<String> = uniq.iter().map(|t| format!("<action type=\"discard\" tile=\"{t}\" />")).collect();
        let n_melds = self.melds[pid].len();
        if drawn.is_some() && !self.any_call && self.discard_count[pid] == 0 && kyuushu_kinds(hand) >= 9 {
            actions.push("<action type=\"kyuushu\" />".to_string());
        }
        if self.is_closed(pid) && self.points[pid] >= 1000 && drawn.is_some() && self.wall.len() >= RIICHI_MIN_WALL
            && self.shanten_of(hand, n_melds) <= 0
        {
            for t in &uniq {
                let tt = norm(parse(t).unwrap());
                let mut rest = hand.clone();
                if let Some(pos) = rest.iter().position(|&x| x == tt) {
                    rest.remove(pos);
                }
                if self.shanten_of(&rest, n_melds) == 0 {
                    actions.push(format!("<action type=\"riichi\" tile=\"{t}\" />"));
                }
            }
        }
        let mut uniq_t: Vec<Tile> = Vec::new();
        for &t in hand {
            if !uniq_t.contains(&t) {
                uniq_t.push(t);
            }
        }
        uniq_t.sort_by_key(|&t| sort_key(t));
        for t in uniq_t {
            if self.can_ankan(pid, t) || self.can_shouminkan(pid, t) {
                actions.push(format!("<action type=\"kan\" tile=\"{}\" />", to_string(t)));
            }
        }
        if let Some(d) = drawn {
            if self.win_result(pid, d, true, false).is_some() {
                actions.push("<action type=\"tsumo\" />".to_string());
            }
        }
        actions
    }

    fn kokushi_robbers(&self, pid: usize, tile: Tile) -> Vec<usize> {
        if !is_orphan(tile) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for p in 0..4 {
            if p == pid || self.is_furiten(p) {
                continue;
            }
            let hand = &self.hands[p];
            if hand.len() != 13 || !self.melds[p].is_empty() {
                continue;
            }
            if !hand.iter().all(|&t| is_orphan(t)) {
                continue;
            }
            if let Some(res) = self.win_result(p, tile, false, true) {
                if res.yaku.iter().any(|y| y.name().contains("Kokushi")) {
                    out.push(p);
                }
            }
        }
        out
    }

    fn chi_pairs(&self, pid: usize, tile: Tile) -> Vec<[Tile; 2]> {
        let v = rank(tile) as i32;
        let base = norm(tile) / 9 * 9;
        let hand = &self.hands[pid];
        let mut pairs = Vec::new();
        for (a, b) in [(v - 2, v - 1), (v - 1, v + 1), (v + 1, v + 2)] {
            if (1..=9).contains(&a) && (1..=9).contains(&b) {
                let ta = base + (a - 1) as u8;
                let tb = base + (b - 1) as u8;
                if hand.contains(&ta) && hand.contains(&tb) {
                    pairs.push([ta, tb]);
                }
            }
        }
        pairs
    }

    pub fn get_interrupt_actions(&self, pid: usize) -> Vec<String> {
        let skip = "<action type=\"skip\" />".to_string();
        let ron = "<action type=\"ron\" />".to_string();
        if let Some(pk) = &self.pending_kan {
            let mut actions = vec![skip];
            if pid != pk.player {
                if pk.ankan {
                    if self.kokushi_robbers(pk.player, pk.tile).contains(&pid) {
                        actions.push(ron);
                    }
                } else if self.can_ron(pid, pk.tile, true) {
                    actions.push(ron);
                }
            }
            return actions;
        }
        let Some(tile) = self.last_discard else { return vec![skip] };
        if self.finished {
            return vec![skip];
        }
        let mut actions = vec![skip];
        if self.can_ron(pid, tile, false) {
            actions.push(ron);
        }
        if self.riichi[pid] || self.wall.is_empty() || self.melds[pid].len() >= MAX_MELDS {
            return actions;
        }
        let cnt = self.hands[pid].iter().filter(|&&t| t == tile).count();
        if cnt >= 2 {
            actions.push(format!("<action type=\"pon\" tile=\"{}\" />", to_string(tile)));
        }
        if cnt >= 3 && self.kan_count < MAX_KANS {
            actions.push(format!("<action type=\"kan\" tile=\"{}\" />", to_string(tile)));
        }
        if let Some(ld) = self.last_discarder {
            if (ld + 1) % 4 == pid && !is_honor(tile) {
                for pair in self.chi_pairs(pid, tile) {
                    actions.push(format!("<action type=\"chi\" tile=\"{}\" with=\"{} {}\" />",
                                         to_string(tile), to_string(pair[0]), to_string(pair[1])));
                }
            }
        }
        actions
    }

    // ---- turn phase ----
    fn do_discard(&mut self, pid: usize, tile: Tile, riichi_mark: bool, red: bool) {
        let tsumogiri = self.last_drawn[pid] == Some(tile) && self.last_drawn_red[pid] == red;
        let sp: Tile = if red { match suit(tile) { 0 => RED_M, 1 => RED_P, _ => RED_S } } else { tile };
        self.river_events[pid].push(RiverEvent { tile: sp, tsumogiri, riichi: riichi_mark, called: false, idx: self.discard_count[pid] });
        if riichi_mark {
            self.riichi_turn[pid] = Some(self.discard_count[pid]);
        }
        self.take_from_hand(pid, tile, 1, red);
        self.discards[pid].push((sp, riichi_mark));
        self.furiten_river[pid].push(tile);
        self.last_discard = Some(tile);
        self.last_discard_red = red;
        self.last_discarder = Some(pid);
        self.last_drawn[pid] = None;
        self.rinshan[pid] = false;
        self.ippatsu[pid] = false;
        self.discard_count[pid] += 1;
        self.kuikae = None;
        while self.pending_dora_reveal > 0 {
            self.pending_dora_reveal -= 1;
            self.reveal_kan_dora();
        }
        if !self.any_call && self.discard_count.iter().sum::<usize>() == 4 && self.discard_count.iter().all(|&c| c == 1) {
            let f0 = self.furiten_river[0][0];
            if (1..4).all(|i| self.furiten_river[i][0] == f0) && (27..=30).contains(&f0) {
                self.pending_abort = Some("四风连打".to_string());
            }
        }
        if self.kan_count >= 4 && self.kan_players.iter().filter(|&&b| b).count() >= 2 {
            self.pending_abort = Some("四杠散了".to_string());
        }
        self.ron_chance = [false; 4];
        for p in 0..4 {
            if p != pid && self.waits(p).contains(&tile) {
                self.ron_chance[p] = true;
            }
        }
    }

    fn forced_discard(&mut self, pid: usize) {
        if self.hands[pid].is_empty() {
            return;
        }
        if self.riichi[pid] {
            if let Some(d) = self.last_drawn[pid] {
                if self.hands[pid].contains(&d) {
                    let red = self.last_drawn_red[pid];
                    self.do_discard(pid, d, false, red);
                    return;
                }
            }
        }
        let banned = self.forbidden_discards(pid);
        let pool: Vec<Tile> = self.hands[pid].iter().copied().filter(|t| !banned.contains(t)).collect();
        let pool = if pool.is_empty() { self.hands[pid].clone() } else { pool };
        let t = pool[self.rng.randbelow(pool.len() as u64) as usize];
        let red = self.plain_copies(pid, t) == 0;
        self.do_discard(pid, t, false, red);
    }

    /// Returns (rewards, done, info). Text observations are never produced.
    pub fn step(&mut self, pid: usize, action_xml: &str) -> ([f64; 4], bool, StepInfo) {
        let mut rewards = [0.0f64; 4];
        let mut info = StepInfo::default();
        let parsed = parse_action(action_xml);
        match parsed {
            None => {
                rewards[pid] = FORMAT_PENALTY;
                self.forced_discard(pid);
                info.discarded = true;
            }
            Some(a) => {
                let drawn = self.last_drawn[pid];
                let banned = self.forbidden_discards(pid);
                let raw_tile = a.tile.as_deref().and_then(parse);
                let red = raw_tile.map(is_red).unwrap_or(false);
                let tile = raw_tile.map(norm);
                let has_copy = match tile {
                    Some(t) => self.hands[pid].contains(&t) && if red {
                        self.red[pid][suit(t) as usize] > 0
                    } else {
                        self.plain_copies(pid, t) > 0
                    },
                    None => false,
                };
                let kind = a.kind.as_str();
                if kind == "discard" && has_copy && !banned.contains(&tile.unwrap())
                    && (!self.riichi[pid] || (tile == drawn && red == self.last_drawn_red[pid]))
                {
                    self.do_discard(pid, tile.unwrap(), false, red);
                    info.discarded = true;
                } else if kind == "riichi" && has_copy && !banned.contains(&tile.unwrap()) && !self.riichi[pid]
                    && self.is_closed(pid) && self.points[pid] >= 1000 && self.wall.len() >= RIICHI_MIN_WALL
                    && self.riichi_tenpai(pid, tile.unwrap())
                {
                    self.riichi[pid] = true;
                    self.riichi_pending = Some(pid);
                    self.daburu[pid] = self.discard_count[pid] == 0 && !self.any_call;
                    self.do_discard(pid, tile.unwrap(), true, red);
                    self.ippatsu[pid] = true;
                    info.discarded = true;
                } else if kind == "kyuushu" && drawn.is_some() && !self.any_call && self.discard_count[pid] == 0
                    && kyuushu_kinds(&self.hands[pid]) >= 9
                {
                    self.abort("九种九牌");
                    return (rewards, true, info);
                } else if kind == "tsumo" && drawn.is_some() && self.win_result(pid, drawn.unwrap(), true, false).is_some() {
                    let res = self.win_result(pid, drawn.unwrap(), true, false).unwrap();
                    self.settle_tsumo(pid, &res);
                    return (rewards, true, info);
                } else if kind == "kan" && tile.is_some() && self.can_ankan(pid, tile.unwrap()) {
                    let t = tile.unwrap();
                    let robbers = self.kokushi_robbers(pid, t);
                    if !robbers.is_empty() {
                        self.pending_kan = Some(PendingKan { player: pid, tile: t, ankan: true, red: false });
                        self.ron_chance = [false; 4];
                        for r in robbers {
                            self.ron_chance[r] = true;
                        }
                        info.chankan = Some(t);
                        return (rewards, false, info);
                    }
                    self.do_ankan(pid, t);
                } else if kind == "kan" && tile.is_some() && self.can_shouminkan(pid, tile.unwrap()) {
                    let t = tile.unwrap();
                    let red_k = self.plain_copies(pid, t) == 0 && self.red[pid][suit(t) as usize] > 0 && !is_honor(t);
                    self.pending_kan = Some(PendingKan { player: pid, tile: t, ankan: false, red: red_k });
                    self.ron_chance = [false; 4];
                    for p in 0..4 {
                        if p != pid && self.waits(p).contains(&t) {
                            self.ron_chance[p] = true;
                        }
                    }
                    info.chankan = Some(t);
                    return (rewards, false, info);
                } else {
                    rewards[pid] = ILLEGAL_PENALTY;
                    self.forced_discard(pid);
                    info.discarded = true;
                }
            }
        }
        sort_hand(&mut self.hands[pid]);
        (rewards, self.finished, info)
    }

    fn riichi_tenpai(&self, pid: usize, discard: Tile) -> bool {
        let mut rest = self.hands[pid].clone();
        if let Some(pos) = rest.iter().position(|&t| t == discard) {
            rest.remove(pos);
        }
        self.shanten_of(&rest, self.melds[pid].len()) == 0
    }

    fn do_ankan(&mut self, pid: usize, tile: Tile) {
        let used_red = self.take_from_hand(pid, tile, 4, false);
        self.melds[pid].push(Meld { kind: MeldKind::Ankan, tiles: vec![tile; 4], opened: false, from: None, red: used_red });
        self.after_kan(pid, true);
    }

    pub fn resolve_pending_kan(&mut self) {
        let Some(pk) = self.pending_kan.take() else { return };
        self.apply_missed_ron();
        if pk.ankan {
            self.do_ankan(pk.player, pk.tile);
            return;
        }
        let pid = pk.player;
        let tile = pk.tile;
        let Some(idx) = self.melds[pid].iter().position(|m| m.kind == MeldKind::Pon && m.tiles[0] == tile) else { return };
        if !self.hands[pid].contains(&tile) {
            return;
        }
        let used = self.take_from_hand(pid, tile, 1, false);
        let m = &mut self.melds[pid][idx];
        m.red += used;
        m.kind = MeldKind::Shouminkan;
        m.tiles = vec![tile; 4];
        self.after_kan(pid, false);
    }

    fn after_kan(&mut self, pid: usize, reveal_now: bool) {
        self.kan_count += 1;
        self.kan_players[pid] = true;
        self.any_call = true;
        self.ippatsu = [false; 4];
        if reveal_now {
            self.reveal_kan_dora();
        } else {
            self.pending_dora_reveal += 1;
        }
        self.rinshan_draw(pid);
        sort_hand(&mut self.hands[pid]);
    }

    fn reveal_kan_dora(&mut self) {
        let k = self.dora_indicators.len();
        if k < 5 {
            self.dora_indicators.push(self.dead_wall[4 + k]);
            self.ura_indicators.push(self.dead_wall[9 + k]);
        }
    }

    fn rinshan_draw(&mut self, pid: usize) {
        if self.rinshan_idx >= 4 || self.wall.is_empty() {
            return;
        }
        let raw = self.dead_wall[self.rinshan_idx];
        self.rinshan_idx += 1;
        self.wall.remove(0);
        let t = self.give(pid, raw);
        sort_hand(&mut self.hands[pid]);
        self.last_drawn[pid] = Some(t);
        self.rinshan[pid] = true;
        self.temp_furiten[pid] = false;
    }

    /// Returns `done`.
    pub fn advance_turn(&mut self) -> bool {
        self.confirm_riichi();
        self.apply_missed_ron();
        if self.finished {
            return true;
        }
        if let Some(reason) = self.pending_abort.take() {
            self.abort(&reason);
            return true;
        }
        self.turn = (self.turn + 1) % 4;
        if self.wall.is_empty() {
            self.ryuukyoku();
            return true;
        }
        let raw = self.wall.pop().unwrap();
        let t = self.give(self.turn, raw);
        let turn = self.turn;
        sort_hand(&mut self.hands[turn]);
        self.last_drawn[turn] = Some(t);
        self.rinshan[turn] = false;
        self.temp_furiten[turn] = false;
        false
    }

    fn confirm_riichi(&mut self) {
        let Some(pid) = self.riichi_pending else { return };
        self.points[pid] -= 1000;
        self.kyotaku += 1000;
        self.riichi_pending = None;
        if self.riichi.iter().all(|&r| r) {
            self.abort("四家立直");
        }
    }

    fn void_riichi(&mut self) {
        let Some(pid) = self.riichi_pending else { return };
        self.riichi[pid] = false;
        self.ippatsu[pid] = false;
        self.daburu[pid] = false;
        self.riichi_pending = None;
        if let Some(last) = self.discards[pid].last_mut() {
            if last.1 {
                last.1 = false;
            }
        }
        if let Some(ev) = self.river_events[pid].last_mut() {
            ev.riichi = false;
        }
        self.riichi_turn[pid] = None;
    }

    fn apply_missed_ron(&mut self) {
        for pid in 0..4 {
            if self.ron_chance[pid] {
                if self.riichi[pid] {
                    self.perm_furiten[pid] = true;
                } else {
                    self.temp_furiten[pid] = true;
                }
            }
        }
        self.ron_chance = [false; 4];
    }

    // ---- interrupt phase ----
    pub fn step_ron(&mut self, player_ids: &[usize]) -> ([f64; 4], bool, StepInfo) {
        let mut rewards = [0.0f64; 4];
        let mut info = StepInfo::default();
        let (tile, discarder, chankan) = if let Some(pk) = &self.pending_kan {
            (pk.tile, pk.player, true)
        } else if let Some(t) = self.last_discard {
            (t, self.last_discarder.unwrap(), false)
        } else {
            for &pid in player_ids {
                rewards[pid] = ILLEGAL_PENALTY;
            }
            return (rewards, false, info);
        };
        let robbers = match &self.pending_kan {
            Some(pk) if pk.ankan => Some(self.kokushi_robbers(discarder, tile)),
            _ => None,
        };
        let mut winners: Vec<(usize, ScoreResult)> = Vec::new();
        for &pid in player_ids {
            let mut result = if !self.is_furiten(pid) { self.win_result(pid, tile, false, chankan) } else { None };
            if let Some(r) = &robbers {
                if !r.contains(&pid) {
                    result = None;
                }
            }
            match result {
                None => rewards[pid] = ILLEGAL_PENALTY,
                Some(res) => winners.push((pid, res)),
            }
        }
        if winners.is_empty() {
            return (rewards, false, info);
        }
        self.pending_kan = None;
        info.interrupt = true;
        info.winners = winners.iter().map(|w| w.0).collect();
        if winners.len() >= 3 {
            if self.riichi_pending == Some(discarder) {
                self.void_riichi();
            }
            self.abort("三家和了");
            info.abort = Some("三家和了".to_string());
            return (rewards, true, info);
        }
        self.settle_ron(&winners, discarder, chankan);
        (rewards, true, info)
    }

    pub fn step_interrupt(&mut self, pid: usize, action_xml: &str) -> ([f64; 4], bool, StepInfo) {
        let mut rewards = [0.0f64; 4];
        let mut info = StepInfo::default();
        let Some(a) = parse_action(action_xml) else {
            rewards[pid] = FORMAT_PENALTY;
            return (rewards, false, info);
        };
        if a.kind == "ron" {
            let (r, done, i) = self.step_ron(&[pid]);
            info.interrupt = i.interrupt;
            return (r, done, info);
        }
        if self.pending_kan.is_some() {
            if a.kind != "skip" {
                rewards[pid] = ILLEGAL_PENALTY;
            }
            return (rewards, false, info);
        }
        let Some(tile) = self.last_discard else { return (rewards, false, info) };
        if a.kind == "skip" {
            return (rewards, false, info);
        }
        let mut interrupted = false;
        let can_meld = !self.riichi[pid] && self.melds[pid].len() < MAX_MELDS && !self.wall.is_empty();
        let cnt = self.hands[pid].iter().filter(|&&t| t == tile).count();
        let ld = self.last_discarder.unwrap();
        if a.kind == "pon" && can_meld && cnt >= 2 {
            let used = self.take_from_hand(pid, tile, 2, false);
            self.melds[pid].push(Meld { kind: MeldKind::Pon, tiles: vec![tile; 3], opened: true, from: Some(ld),
                                        red: used + self.last_discard_red as u8 });
            self.record_pao(pid, tile, Some(ld));
            self.kuikae = Some((pid, self.kuikae_tiles(tile, &[])));
            self.claim_discard(pid);
            interrupted = true;
        } else if a.kind == "kan" && can_meld && cnt >= 3 && self.kan_count < MAX_KANS {
            let used = self.take_from_hand(pid, tile, 3, false);
            self.melds[pid].push(Meld { kind: MeldKind::Kan, tiles: vec![tile; 4], opened: true, from: Some(ld),
                                        red: used + self.last_discard_red as u8 });
            self.record_pao(pid, tile, Some(ld));
            self.claim_discard(pid);
            self.after_kan(pid, false);
            interrupted = true;
        } else if a.kind == "chi" && can_meld && !is_honor(tile) && (ld + 1) % 4 == pid {
            let pairs = self.chi_pairs(pid, tile);
            if !pairs.is_empty() {
                let wanted: Vec<Tile> = a.with.as_deref().unwrap_or("").split_whitespace()
                    .filter_map(parse).map(norm).collect();
                let mut ws = wanted.clone();
                ws.sort();
                let chosen = pairs.iter().find(|p| { let mut q = p.to_vec(); q.sort(); q == ws }).copied()
                    .unwrap_or(pairs[0]);
                let mut used = 0u8;
                for &t in &chosen {
                    used += self.take_from_hand(pid, t, 1, false);
                }
                let mut seq = vec![chosen[0], chosen[1], tile];
                seq.sort_by_key(|&t| sort_key(t));
                self.melds[pid].push(Meld { kind: MeldKind::Chi, tiles: seq, opened: true, from: Some(ld),
                                            red: used + self.last_discard_red as u8 });
                self.kuikae = Some((pid, self.kuikae_tiles(tile, &chosen)));
                self.claim_discard(pid);
                interrupted = true;
            }
        }
        if !interrupted {
            rewards[pid] = ILLEGAL_PENALTY;
        }
        sort_hand(&mut self.hands[pid]);
        info.interrupt = interrupted;
        (rewards, false, info)
    }

    fn claim_discard(&mut self, pid: usize) {
        self.pending_abort = None;
        self.confirm_riichi();
        self.apply_missed_ron();
        let ld = self.last_discarder.unwrap();
        self.discards[ld].pop();
        if let Some(ev) = self.river_events[ld].last_mut() {
            ev.called = true;
        }
        self.turn = pid;
        self.last_discard = None;
        self.last_drawn[pid] = None;
        self.rinshan[pid] = false;
        self.any_call = true;
        self.ippatsu = [false; 4];
        self.temp_furiten[pid] = false;
    }

    fn record_pao(&mut self, pid: usize, tile: Tile, feeder: Option<usize>) {
        let Some(f) = feeder else { return };
        if f == pid {
            return;
        }
        let n = norm(tile);
        let group: &[u8] = if (31..=33).contains(&n) { &[31, 32, 33] } else if (27..=30).contains(&n) { &[27, 28, 29, 30] } else { return };
        let mut owned = Vec::new();
        for m in &self.melds[pid] {
            let t = m.tiles[0];
            if group.contains(&t) && !owned.contains(&t) {
                owned.push(t);
            }
        }
        if owned.len() == group.len() {
            self.pao[pid] = Some(f);
        }
    }

    // ---- settlement ----
    fn pao_liable(&self, winner: usize, result: &ScoreResult) -> Option<usize> {
        let liable = self.pao[winner]?;
        if result.yaku.iter().any(|y| matches!(y, score::Yaku::Daisangen | score::Yaku::Daisuushi)) {
            Some(liable)
        } else {
            None
        }
    }

    fn yaku_str(result: &ScoreResult) -> String {
        result.yaku.iter().map(|y| y.name()).collect::<Vec<_>>().join(", ")
    }

    fn settle_tsumo(&mut self, winner: usize, result: &ScoreResult) {
        let main = result.main;
        let additional = result.additional;
        let liable = self.pao_liable(winner, result);
        let total = if winner == self.dealer { main * 3 } else { main + additional * 2 };
        if let Some(l) = liable {
            self.points[l] -= total;
            self.points[winner] += total;
        } else if winner == self.dealer {
            for i in 0..4 {
                if i != winner {
                    self.points[i] -= main;
                    self.points[winner] += main;
                }
            }
        } else {
            for i in 0..4 {
                if i == winner {
                    continue;
                }
                let pay = if i == self.dealer { main } else { additional };
                self.points[i] -= pay;
                self.points[winner] += pay;
            }
        }
        self.points[winner] += self.kyotaku;
        self.kyotaku = 0;
        self.finished = true;
        let pao = liable.map(|l| format!(" | 包牌:玩家{l}")).unwrap_or_default();
        self.result_summary = format!("玩家{} 自摸 | {}番{}符 | {}{} | 点数: {}", winner, result.han, result.fu,
                                      Self::yaku_str(result), pao, fmt_points(&self.points));
        self.compute_final_rewards(None);
    }

    fn settle_ron(&mut self, winners: &[(usize, ScoreResult)], discarder: usize, chankan: bool) {
        if self.riichi_pending == Some(discarder) {
            self.void_riichi();
        } else {
            self.confirm_riichi();
        }
        let mut parts = Vec::new();
        for (pid, result) in winners {
            let cost = result.main;
            let liable = self.pao_liable(*pid, result);
            match liable {
                Some(l) if l != discarder => {
                    let half = cost / 2;
                    self.points[l] -= half;
                    self.points[discarder] -= cost - half;
                }
                _ => self.points[discarder] -= cost,
            }
            self.points[*pid] += cost;
            let how = if chankan { "抢杠" } else { "荣和" };
            let pao = liable.map(|l| format!(" | 包牌:玩家{l}")).unwrap_or_default();
            parts.push(format!("玩家{} {}(放铳:玩家{}) | {}番{}符 | {}{}", pid, how, discarder, result.han, result.fu,
                               Self::yaku_str(result), pao));
        }
        self.points[winners[0].0] += self.kyotaku;
        self.kyotaku = 0;
        self.finished = true;
        self.result_summary = format!("{}{} | 点数: {}", parts.join(" ; "),
                                      if winners.len() > 1 { " | 双响" } else { "" }, fmt_points(&self.points));
        self.compute_final_rewards(Some(discarder));
    }

    fn abort(&mut self, reason: &str) {
        if self.finished {
            return;
        }
        self.pending_abort = None;
        self.finished = true;
        self.result_summary = format!("途中流局({}) | 点数: {}", reason, fmt_points(&self.points));
        self.compute_final_rewards(None);
    }

    fn nagashi_mangan(&self) -> Vec<usize> {
        (0..4).filter(|&i| {
            let ev = &self.river_events[i];
            !ev.is_empty() && ev.iter().all(|e| !is_red(e.tile) && is_orphan(e.tile) && !e.called)
        }).collect()
    }

    fn ryuukyoku(&mut self) {
        if self.finished {
            return;
        }
        let nagashi = self.nagashi_mangan();
        if !nagashi.is_empty() {
            for &w in &nagashi {
                for i in 0..4 {
                    if i == w {
                        continue;
                    }
                    let pay = if w == self.dealer || i == self.dealer { 4000 } else { 2000 };
                    self.points[i] -= pay;
                    self.points[w] += pay;
                }
            }
            self.finished = true;
            self.result_summary = format!("流局满贯 | 玩家{} | 点数: {}", fmt_list(&nagashi), fmt_points(&self.points));
            self.compute_final_rewards(None);
            return;
        }
        let tenpai: Vec<bool> = (0..4).map(|i| self.shanten_of(&self.hands[i], self.melds[i].len()) <= 0).collect();
        let n = tenpai.iter().filter(|&&b| b).count() as i64;
        if n > 0 && n < 4 {
            let gain = 3000 / n;
            let loss = 3000 / (4 - n);
            for i in 0..4 {
                self.points[i] += if tenpai[i] { gain } else { -loss };
            }
        }
        self.finished = true;
        let tl: Vec<usize> = (0..4).filter(|&i| tenpai[i]).collect();
        self.result_summary = format!("流局 | 听牌: {} | 点数: {}", fmt_list(&tl), fmt_points(&self.points));
        self.compute_final_rewards(None);
    }

    fn compute_final_rewards(&mut self, _houjuu: Option<usize>) {
        let mut fr = [0.0f64; 4];
        for i in 0..4 {
            fr[i] = (self.points[i] - self.start_points[i]) as f64 * REWARD_SCALE;
        }
        // placement bonus with ties sharing the average of the spanned positions
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|&i| -self.points[i]);      // stable, like Python's sorted
        let mut pos = 0;
        while pos < 4 {
            let p = self.points[order[pos]];
            let tied: Vec<usize> = order.iter().copied().filter(|&i| self.points[i] == p).collect();
            let shared: f64 = RANK_BONUS[pos..pos + tied.len()].iter().sum::<f64>() / tied.len() as f64;
            for &i in &tied {
                fr[i] += shared;
            }
            pos += tied.len();
        }
        self.final_rewards = Some(fr);
    }
}

/// claims.py::_resolve_claims — one interrupt window. `cands` = (seat, action xml)
/// in seat order from the discarder. Returns (executed seats, done, reward per cand).
pub fn resolve_claims(t: &mut Table, cands: &[(usize, String)]) -> (Vec<usize>, bool, Vec<f64>) {
    let mut rewards = vec![0.0f64; cands.len()];
    let mut kinds: Vec<Option<String>> = Vec::with_capacity(cands.len());
    for (i, (_, xml)) in cands.iter().enumerate() {
        match parse_action(xml) {
            None => {
                rewards[i] = FORMAT_PENALTY;
                kinds.push(None);
            }
            Some(a) => kinds.push(Some(a.kind)),
        }
    }
    let ron_idx: Vec<usize> = (0..cands.len()).filter(|&i| kinds[i].as_deref() == Some("ron")).collect();
    if !ron_idx.is_empty() {
        let ids: Vec<usize> = ron_idx.iter().map(|&i| cands[i].0).collect();
        let (rw, done, info) = t.step_ron(&ids);
        for &i in &ron_idx {
            rewards[i] = rw[cands[i].0];
        }
        if !info.winners.is_empty() {
            let executed: Vec<usize> = ron_idx.iter().map(|&i| cands[i].0).filter(|p| info.winners.contains(p)).collect();
            return (executed, done, rewards);
        }
    }
    let prio = |k: &str| match k { "kan" => 1, "pon" => 2, "chi" => 3, _ => 9 };
    let mut others: Vec<usize> = (0..cands.len())
        .filter(|&i| matches!(kinds[i].as_deref(), Some(k) if k != "ron" && k != "skip"))
        .collect();
    others.sort_by_key(|&i| prio(kinds[i].as_deref().unwrap()));   // stable, like Python
    let mut executed = Vec::new();
    for i in others {
        if !executed.is_empty() {
            continue;
        }
        let (rw, _done, info) = t.step_interrupt(cands[i].0, &cands[i].1);
        rewards[i] = rw[cands[i].0];
        if info.interrupt {
            executed.push(cands[i].0);
        }
    }
    (executed, false, rewards)
}
