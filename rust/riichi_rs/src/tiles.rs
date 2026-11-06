//! Tile codes. 0..=33 = normalized 34-index (m 0-8, p 9-17, s 18-26, z 27-33);
//! 34/35/36 = the red fives 0m/0p/0s (normalize to 4/13/22). Strings follow the
//! Python engine's spelling ("5m", "0p", "1z").
pub type Tile = u8;

pub const RED_M: Tile = 34;
pub const RED_P: Tile = 35;
pub const RED_S: Tile = 36;

#[inline]
pub fn is_red(t: Tile) -> bool {
    t >= 34
}

/// Red five -> plain five; everything else unchanged (Python `norm_tile`).
#[inline]
pub fn norm(t: Tile) -> Tile {
    match t {
        RED_M => 4,
        RED_P => 13,
        RED_S => 22,
        _ => t,
    }
}

#[inline]
pub fn suit(t: Tile) -> u8 {
    // 0 m, 1 p, 2 s, 3 z  (of the normalized index)
    (norm(t) / 9) as u8
}

#[inline]
pub fn rank(t: Tile) -> u8 {
    // 1..9 for suits, 1..7 for honors
    (norm(t) % 9) as u8 + 1
}

#[inline]
pub fn is_honor(t: Tile) -> bool {
    norm(t) >= 27
}

#[inline]
pub fn is_terminal_or_honor(t: Tile) -> bool {
    let n = norm(t);
    n >= 27 || n % 9 == 0 || n % 9 == 8
}

/// Python `sort_key`: suit order p < s < m < z, then rank.
#[inline]
pub fn sort_key(t: Tile) -> (u8, u8) {
    let order = match suit(t) {
        1 => 0, // p
        2 => 1, // s
        0 => 2, // m
        _ => 3, // z
    };
    (order, rank(t))
}

pub fn sort_hand(hand: &mut Vec<Tile>) {
    hand.sort_by_key(|&t| sort_key(t));
}

pub fn to_string(t: Tile) -> String {
    if is_red(t) {
        return match t {
            RED_M => "0m".into(),
            RED_P => "0p".into(),
            _ => "0s".into(),
        };
    }
    let s = match t / 9 {
        0 => 'm',
        1 => 'p',
        2 => 's',
        _ => 'z',
    };
    format!("{}{}", t % 9 + 1, s)
}

/// Parse "5m" / "0p" / "1z" (a trailing '*' riichi mark is ignored).
pub fn parse(s: &str) -> Option<Tile> {
    let b = s.trim_end_matches('*').as_bytes();
    if b.len() != 2 {
        return None;
    }
    let d = b[0].wrapping_sub(b'0');
    let base = match b[1] {
        b'm' => 0u8,
        b'p' => 9,
        b's' => 18,
        b'z' => 27,
        _ => return None,
    };
    if d == 0 {
        return match b[1] {
            b'm' => Some(RED_M),
            b'p' => Some(RED_P),
            b's' => Some(RED_S),
            _ => None,
        };
    }
    if d > 9 || (base == 27 && d > 7) {
        return None;
    }
    Some(base + d - 1)
}

/// Dora tile pointed to by an indicator (Python `dora_from_indicator`).
pub fn dora_from_indicator(ind: Tile) -> Tile {
    let n = norm(ind);
    if n >= 27 {
        let r = n - 27; // 0..6
        if r <= 3 {
            27 + (r + 1) % 4
        } else {
            27 + 4 + (r - 4 + 1) % 3
        }
    } else {
        let base = n / 9 * 9;
        base + (n - base + 1) % 9
    }
}
