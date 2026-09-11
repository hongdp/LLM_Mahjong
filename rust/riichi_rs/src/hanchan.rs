//! hanchan.py::MatchState — between-deal match bookkeeping (renchan, honba,
//! kyotaku carry, bust, placements, uma), ported 1:1 from the Python driver
//! (post 2026-09-11 fixes: engine-settled 流し満貫 is not paid again; the
//! driver's point delta is the per-deal reward).
use crate::table::Table;

pub const UMA: [i64; 4] = [15000, 5000, -5000, -15000];

#[derive(Clone, Debug)]
pub struct DealFact {
    pub result: String,
    pub riichi: [bool; 4],
    pub n_melds: [usize; 4],
    pub n_discards: usize,
    pub points: [i64; 4],
    pub start_points: [i64; 4],
}

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub final_points: [i64; 4],
    pub placements: [usize; 4],   // 1..4 per seat
    pub uma_points: [i64; 4],     // final - 25000 + UMA[placement-1]
    pub busted: bool,
    pub n_deals: usize,
}

#[derive(Clone, Debug)]
pub struct MatchState {
    pub points: [i64; 4],
    pub dealer: usize,
    pub rw: usize,
    pub honba: i64,
    pub kyotaku: i64,
    pub start_dealer: usize,
    pub n: usize,
    pub max_deals: usize,
    pub done: bool,
    pub busted: bool,
}

/// Seats in placement order; ties -> closer to the starting East wins.
pub fn rank_order(points: &[i64; 4], start_dealer: usize) -> [usize; 4] {
    let mut order = [0usize, 1, 2, 3];
    order.sort_by_key(|&s| (-points[s], (s + 4 - start_dealer) % 4));
    order
}

impl MatchState {
    pub fn new(max_deals: usize) -> Self {
        MatchState { points: [25000; 4], dealer: 0, rw: 0, honba: 0, kyotaku: 0, start_dealer: 0, n: 0, max_deals, done: false, busted: false }
    }

    /// Context for the next deal: (dealer, round_wind_idx, points, kyotaku).
    pub fn begin_deal(&mut self) -> (usize, usize, [i64; 4], i64) {
        self.n += 1;
        (self.dealer, self.rw, self.points, self.kyotaku)
    }

    /// Consume a finished deal's table and advance the match.
    pub fn settle(&mut self, t: &Table) {
        let de = t.deal_end.clone().unwrap_or_default();
        let dealer = self.dealer;
        let honba = self.honba;
        let mut points = t.points;
        // driver-side honba payments (the engine has no honba concept)
        if !de.winners.is_empty() && honba > 0 {
            let mut seen: Vec<usize> = Vec::new();
            for &w in &de.winners {
                if seen.contains(&w) {
                    continue;
                }
                seen.push(w);
                if let Some(l) = de.houjuu {
                    points[w] += 300 * honba;
                    points[l] -= 300 * honba;
                } else {
                    for p in 0..4 {
                        if p != w {
                            points[w] += 100 * honba;
                            points[p] -= 100 * honba;
                        }
                    }
                }
            }
        }
        self.kyotaku = if de.winners.is_empty() { t.kyotaku } else { 0 };
        let dealer_won = de.winners.contains(&dealer);
        let is_draw = de.winners.is_empty();
        let is_abort = de.abort;
        let mut dealer_tenpai_at_draw = is_abort;
        if is_draw && de.nagashi {
            dealer_tenpai_at_draw = t.shanten_of(&t.hands[dealer], t.melds[dealer].len()) <= 0;
        } else if is_draw && !is_abort {
            if let Some(tp) = de.tenpai {
                dealer_tenpai_at_draw = tp[dealer];
            }
        }
        self.points = points;
        if points.iter().any(|&p| p < 0) {
            self.busted = true;
            self.done = true;
            return;
        }
        if dealer_won || (is_draw && dealer_tenpai_at_draw) {
            self.honba += 1;
        } else {
            self.honba = if is_draw { self.honba + 1 } else { 0 };
            self.dealer = (self.dealer + 1) % 4;
            if self.dealer == self.start_dealer {
                if self.rw >= 1 {
                    self.done = true;
                    return;
                }
                self.rw += 1;
            }
        }
        if self.rw >= 2 || self.n >= self.max_deals {
            self.done = true;
        }
    }

    pub fn result(&self) -> MatchResult {
        let mut final_points = self.points;
        if self.kyotaku > 0 {
            let top = rank_order(&final_points, self.start_dealer)[0];
            final_points[top] += self.kyotaku;
        }
        let order = rank_order(&final_points, self.start_dealer);
        let mut placements = [0usize; 4];
        for (rank, &seat) in order.iter().enumerate() {
            placements[seat] = rank + 1;
        }
        let mut uma_points = [0i64; 4];
        for s in 0..4 {
            uma_points[s] = final_points[s] - 25000 + UMA[placements[s] - 1];
        }
        MatchResult { final_points, placements, uma_points, busted: self.busted, n_deals: self.n }
    }
}
