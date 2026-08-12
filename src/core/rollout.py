from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class TrajectoryStep:
    """Represents a single state-action-reward transition.

    gen_token_ids / old_logprobs are populated only when the rollout runs
    with capture_logprobs=True (PPO): the exact sampled token ids and the
    behavior policy's raw-logit logprobs for them. Storing ids (not text)
    sidesteps retokenization drift between rollout and update time.
    """
    prompt_text: str
    action_text: str
    reward: float
    is_terminal: bool
    gen_token_ids: list = None
    old_logprobs: list = None
    # Settlement breakdown, filled on the terminal step only (for logs/UI —
    # training keeps consuming the combined `reward`).
    settlement: float = None      # final_rewards[pid] as merged into reward
    final_points: int = None      # points after the game's point transfers
    rank_bonus: float = None      # placement share of the settlement
    game_result: str = None       # engine result_summary string

class ReplayBuffer:
    """
    Stores full game trajectories (episodes) for all agents.
    Calculates advantages (Return-to-Go) before yielding batches.
    """
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.episodes: List[List[TrajectoryStep]] = []
        # Optional per-episode covariates (e.g. initial-hand quality).
        # When provided for all episodes, returns are corrected by a linear
        # control-variate baseline before normalization: an action-
        # independent function of the episode's STARTING state, so the
        # expected gradient is unchanged while deal-luck variance drops.
        self.covariates: List[float] = []
        # Optional duplicate-deal grouping: episodes sharing a key started
        # from the IDENTICAL wall in the IDENTICAL seat (common random
        # numbers). The leave-one-out group mean is then an unbiased,
        # action-independent estimate of that start state's value, so
        # subtracting it removes deal luck without biasing the gradient.
        # Caveat (measured, not assumed): replicas share the wall and the
        # seat's dealt hand exactly, but a seat's FIRST decision can land at
        # different points across replicas (an interrupt opportunity depends
        # on what others discarded), so G0 is not always taken from an
        # identical state — the shared component is the deal, which is what
        # this cancels —
        # and unlike a learned V(s) it needs nothing to be predictable
        # (exp4 probe: mid-game return is intrinsically ~unpredictable).
        self.group_keys: List = []

    def add_episode(self, episode: List[TrajectoryStep], covariate: float = None,
                    group_key=None):
        self.episodes.append(episode)
        if covariate is not None:
            self.covariates.append(float(covariate))
        if group_key is not None:
            self.group_keys.append(group_key)

    def calculate_advantages_and_flatten(self) -> List[Dict[str, Any]]:
        """
        Calculates Return-To-Go for each step in every episode.
        Normalizes the advantages across the entire buffer (GRPO/PPO style).
        Returns a flat list of training samples.
        """
        all_samples = []
        all_returns = []

        # 1. Calculate Returns (discounted sum of future rewards)
        per_episode_returns = []
        for episode in self.episodes:
            returns = []
            R = 0
            # Iterate backwards
            for step in reversed(episode):
                R = step.reward + self.gamma * R
                returns.insert(0, R)
            per_episode_returns.append(returns)

        # 1b. Optional control-variate baseline: fit G0 ~ a + b*cov across
        # episodes, then spread each episode's correction EVENLY over its
        # per-step rewards before recomputing returns. Subtracting the full
        # episode constant from every G_t would over-correct late steps
        # (their tails contain only part of the luck) and flip the
        # correlation sign — caught by test_covariate_baseline.
        # Caveat (documented): the 1/n spread couples weakly to episode
        # length, which is barely policy-controllable here (wall-bound).
        if (len(self.covariates) == len(self.episodes)
                and len(self.episodes) >= 4):
            covs = torch.tensor(self.covariates, dtype=torch.float64)
            g0 = torch.tensor([r[0] if r else 0.0
                               for r in per_episode_returns],
                              dtype=torch.float64)
            var = covs.var(unbiased=False)
            if var > 1e-8:
                b = ((covs - covs.mean()) * (g0 - g0.mean())).mean() / var
                a = g0.mean() - b * covs.mean()
                for e_idx, returns in enumerate(per_episode_returns):
                    n = len(returns)
                    if n == 0:
                        continue
                    d = (a + b * covs[e_idx]).item() / n   # per-step share
                    corrected = []
                    for t, r in enumerate(returns):
                        # G_t loses the discounted sum of the remaining
                        # per-step shares: d * sum_{k=t}^{n-1} gamma^(k-t)
                        m = n - t
                        tail = m if self.gamma == 1.0 else \
                            (1.0 - self.gamma ** m) / (1.0 - self.gamma)
                        corrected.append(r - d * tail)
                    per_episode_returns[e_idx] = corrected

        # 1c. Duplicate-deal group baseline (preferred over 1b when present):
        # leave-one-out mean of G0 within each (deal, seat) group. Spread
        # over per-step rewards exactly like 1b — subtracting the constant
        # from every G_t would over-correct late steps.
        if (len(self.group_keys) == len(self.episodes)
                and len(self.episodes) >= 2):
            from collections import defaultdict
            groups = defaultdict(list)
            for i, k in enumerate(self.group_keys):
                groups[k].append(i)
            g0 = [r[0] if r else 0.0 for r in per_episode_returns]
            for key, idxs in groups.items():
                if len(idxs) < 2:
                    continue          # no replicas -> no baseline available
                total = sum(g0[i] for i in idxs)
                for i in idxs:
                    loo = (total - g0[i]) / (len(idxs) - 1)   # unbiased
                    returns = per_episode_returns[i]
                    n = len(returns)
                    if n == 0:
                        continue
                    d = loo / n
                    corrected = []
                    for t, r in enumerate(returns):
                        m = n - t
                        tail = m if self.gamma == 1.0 else \
                            (1.0 - self.gamma ** m) / (1.0 - self.gamma)
                        corrected.append(r - d * tail)
                    per_episode_returns[i] = corrected

        for episode, returns in zip(self.episodes, per_episode_returns):
            all_returns.extend(returns)
            # Pair steps with returns
            for step, R in zip(episode, returns):
                all_samples.append({
                    "prompt": step.prompt_text,
                    "action": step.action_text,
                    "return": R,
                    "gen_token_ids": step.gen_token_ids,
                    "old_logprobs": step.old_logprobs,
                })

        if not all_returns:
            return []

        # 2. Normalize Returns into Advantages
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        mean = returns_tensor.mean()
        std = returns_tensor.std() + 1e-8
        advantages = (returns_tensor - mean) / std

        # 3. Inject advantages into samples
        for i, sample in enumerate(all_samples):
            sample["advantage"] = advantages[i].item()

        return all_samples

    def clear(self):
        self.episodes = []
