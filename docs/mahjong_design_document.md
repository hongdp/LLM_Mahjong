# LLM-based Multi-Agent Riichi Mahjong Self-Play and RL System

## 1. Executive Summary
This project aims to build a fully Large Language Model (LLM) based multi-agent game system. Four independent LLM instances will engage in four-player Riichi Mahjong within a Partially Observable Markov Decision Process (POMDP) by maintaining their private contexts.

The project utilizes the Group Relative Policy Optimization (GRPO) algorithm and Self-Play mechanisms to drive the models to evolve from basic tile-efficiency deduction using the Five Block Method, up to high-level game strategies featuring global defense and board-state evaluation.

---

## 2. System Architecture
The entire system is divided into three core layers: the Table Engine Layer, the Agent Orchestration Layer, and the RL Training Layer.

### 2.1 Table Engine Layer
Acting as the "God-view" state machine, this layer is responsible for the absolute execution of the game's physical rules. It is developed in Python 3.10 to ensure seamless integration with the modern deep learning ecosystem.
*   **Match Management**: Handles shuffling, dealing, and turn rotation.
*   **Information Isolation**: Strictly maintains the global state and distributes restricted, localized views (POMDP) to the four Agents on demand.
*   **Legality Validation**: Interrupts illegal actions (e.g., incorrect meld targets, discarding non-existent tiles) and returns strong negative rewards.
*   **Scoring & Settlement Module**: Integrates a custom Mahjong point calculator to provide precise Fu (mini-points) and Han (doubles) calculations upon Ron/Tsumo (winning), serving as the absolute Ground Truth Reward for the round.

### 2.2 Agent Orchestration Layer (LangGraph)
Because Mahjong includes interrupt mechanisms (e.g., after one player discards, the other three can concurrently decide whether to call/meld), traditional linear turn-based systems are insufficient. We use LangGraph to build an asynchronous, event-driven graph structure:
*   **Turn Node**: During the draw-discard phase, the Table Engine wakes up the current turn's Agent node, passing the State containing the newly drawn tile.
*   **Interrupt Node**: After a player discards, the state machine suspends the main loop and runs the other three Agents in parallel with a very short timeout. If a `<pon>` / `<ron>` call response is received, edge routing is executed to branch the graph.

### 2.3 RL Training Layer (GRPO)
We abandon the highly VRAM-intensive traditional PPO (Actor-Critic architecture) in favor of the GRPO algorithm to maximize hardware utilization.
*   **Policy Model (Actor)**: Independently running LLM instances (e.g., Qwen-2.5-1.5B/7B level), undergoing distributed inference and training on TPU/GPU clusters.
*   **Sampling Mechanism**: For every decision point (Prompt), $G$ different `<think>` reasoning trajectories and final actions are generated via temperature sampling. Advantages are calculated within the group to perform policy updates.
*   **Gradient Update Timing (Delayed Reward)**: Model weights are **never** updated after a single step or discard. Mahjong is a game of delayed rewards. The system decouples environment interaction from gradient updates by collecting a batch of complete trajectories (episodes). Step-level heuristics and Round-level outcomes are accumulated, and the model is updated in batches using the calculated group advantage $A_i$. This prevents the policy from collapsing into short-sighted, greedy behaviors (e.g., maximizing immediate Ukeire at the cost of dealing-in later).

---

## 3. State & Action Space Definition

### 3.1 State Representation (Text-based Prompting)
The text State received by the Agent must be highly compressed. We adopt the Tenhou Pinyin Notation, divided into three blocks:
*   **Global**: Round Wind: East, Round Number: 1, Dora Indicator: 3p, Kyoutaku (Riichi Bets): 0
*   **Private**: Seat Wind: South, Points: 25000, Hand: 1m 2m 3m 5p 6p 7p 1s 1s 2s 3s 4s 5s 6s (Forced sorting to align with Five Block Method comprehension).
*   **Public**: Records the discard piles and melds of the other three players in clockwise order.

### 3.2 Action Space Constraints (XML Action Tags)
The model must output its final action using XML tags. A regex parsing module will hard-extract these:
*   **Closed Hand Actions**: `<action type="discard" tile="1s" />`, `<action type="riichi" tile="1s" />`, `<action type="tsumo" />`
*   **Meld/Ron Responses**: `<action type="pon" target="3p" discard="1m" />`, `<action type="ron" />`, `<action type="skip" />`

---

## 4. Reward Shaping
The reward mechanism smoothly transitions from single-step greedy optimization to global end-game optimization.

### 4.1 Step-level Rewards (Real-time Tile Efficiency)
Guides the model to establish correct tile logic fundamentals:
*   **Format/Legality Penalty**: Violating XML formatting or outputting hallucinations ($R = -10.0$).
*   **Ukeire (Tile Acceptance) Reward**: Calculates Ukeire after a discard. If it results in maximum Ukeire, $R = +2.0$; otherwise, penalize proportionally based on the difference.
*   **Shanten (Steps to Tenpai) Reward**: Decreasing Shanten ($R = +5.0$), regressing Shanten ($R = -5.0$).

### 4.2 Round-level Rewards (Round Settlement)
Introduces the game theory of risk and reward:
*   **Ron/Tsumo (Win)**: Utilizes the custom scoring calculator to obtain base points. The reward correlates positively with points (e.g., $R = \text{Base Points} \times 0.001$).
*   **Deal-in (Houjuu) Penalty**: Strong negative feedback forcing the model to learn tile reading and folding (e.g., $R = -(\text{Points Lost} \times 0.001) - 5.0$ extra penalty).

### 4.3 Match-level Rewards (Hanchan Placement)
Introduces Uma (placement bonus) settlement. First place receives a massive positive Advantage, while fourth place suffers a severe gradient penalty, cultivating a strong "avoid 4th place" awareness.

---

## 5. Training Roadmap

### Phase 1: Heuristic Bootstrapping (Single Node)
*   **Goal**: Enable a single LLM to master basic rules and tile efficiency.
*   **Environment**: 1 LLM Agent vs. 3 baseline bots built on pure code logic (e.g., strictly playing max-Ukeire algorithms).
*   **Outcome**: A baseline model capable of stably deducing optimal discards using the Five Block Method, strictly adhering to the XML output format.

### Phase 2: Multi-Agent Self-Play Rollout
*   **Goal**: Emergence of defensive strategies and high-level board evaluation.
*   **Environment**: Deployment of 4 LLM instances (replicas of the same weights).
*   **Mechanism**: Maintains a Model Registry. The latest Actor model does not always play against itself; it plays against earlier checkpoints with a certain probability (e.g., 20%) to prevent Strategy Forgetting (policy collapse).

### Phase 3: Cloud-Scale RL
*   **Deployment**: Migration to Google Cloud. Utilizes Colab Enterprise for experiment tracking, running dozens of Table Engines in parallel on TPU/GPU nodes for high-throughput experience sampling.
*   **Optimization**: At this stage, focus heavily on observing GRPO's intra-group variance, tuning the sample quantity $G$ to strike a balance between sampling costs and convergence speed.
