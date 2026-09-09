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
