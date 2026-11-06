//! exp108: privileged (oracle) critic features, 319-dim, mirroring the Python
//! `src/agents/dnn/oracle_features.py` layout slot for slot (parity test:
//! tests/test_oracle_rust_parity.py). Seen only by a critic, never by the policy.
//!   [0:102)   opponents' concealed hand counts / 4   (3 x 34, relative seats 1..3)
//!   [102:111) opponents' red fives in hand (m, p, s) (3 x 3)
//!   [111:213) opponents' waits (1 = completes)       (3 x 34, 13-tile states only)
//!   [213:216) opponents' shanten: (sh + 1).clamp(0, 6) / 6
//!   [216:250) live wall counts / 4, [250] wall length / 70
//!   [251:285) next kan-dora indicator one-hot (hidden), [285:319) ura under the first dora
use crate::table::Table;
use crate::tiles::norm;

pub const ORACLE_DIM: usize = 319;

pub fn oracle_vec(t: &Table, seat: usize, f: &mut [f32]) {
    debug_assert!(f.len() == ORACLE_DIM);
    for x in f.iter_mut() {
        *x = 0.0;
    }
    for off in 1..4 {
        let opp = (seat + off) % 4;
        let base = (off - 1) * 34;
        for &tile in &t.hands[opp] {
            f[base + norm(tile) as usize] += 0.25;
        }
        for k in 0..3 {
            f[102 + (off - 1) * 3 + k] = t.red[opp][k] as f32;
        }
        if t.hands[opp].len() % 3 == 1 {
            for w in t.waits(opp) {
                f[111 + base + norm(w) as usize] = 1.0;
            }
        }
        let sh = t.shanten_of(&t.hands[opp], t.melds[opp].len());
        f[213 + off - 1] = ((sh + 1).clamp(0, 6)) as f32 / 6.0;
    }
    for &tile in &t.wall {
        f[216 + norm(tile) as usize] += 0.25;
    }
    f[250] = t.wall.len() as f32 / 70.0;
    let nxt = 4 + t.dora_indicators.len();
    if nxt < 9 && nxt < t.dead_wall.len() {
        f[251 + norm(t.dead_wall[nxt]) as usize] = 1.0;
    }
    if t.dead_wall.len() > 9 {
        f[285 + norm(t.dead_wall[9]) as usize] = 1.0;
    }
}
