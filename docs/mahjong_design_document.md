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

### 2.3 RL Training Layer (Custom Replay Buffer & Advantage Policy)
We abandoned Hugging Face's `trl.GRPOTrainer` because it is strictly designed for stateless, single-turn Instruction Tuning (QA) and fundamentally incompatible with interactive, multi-turn POMDPs. Instead, we built a bespoke Advantage-Weighted RL loop:
*   **Asynchronous Replay Buffer (`src/core/rollout.py`)**: Before gradients are computed, the LLM takes control of all 4 agents and plays $N$ full games to completion inside the LangGraph engine. Every `(State, Action, Reward)` is recorded.
*   **Delayed Reward & Advantage Calculation**: Once a game terminates, the Replay Buffer calculates the discounted Return-to-Go for every historical step. A sub-optimal discard on Turn 2 is correctly penalized if it leads to Houjuu (dealing in) on Turn 50.
*   **Custom Training Loop (`src/core/trainer.py`)**: The model computes the Negative Log-Likelihood of the actions it took during the rollout, multiplies it by the calculated Advantage (Policy Gradient / Behavior Cloning), and performs backpropagation. This completely decouples rollout from optimization, matching the AlphaGo paradigm.

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

### Phase 0: Local Architecture Verification & Decoupling (Completed)
*   **Goal**: Ensure the RL loop can perform forward/backward passes sequentially without crashing, securely decouple domain-specific rules (`src/tasks/mahjong`) from the RL engine (`src/core`), and stabilize training against format hallucinations.
*   **Infrastructure**: Implemented `python-dotenv` for local token injection and LangGraph for state routing. Upgraded to a full Python-based 136-tile deck Mahjong simulator. Upgraded LLM prompts to enforce Chain-of-Thought (CoT) reasoning via `<think>` tags and strict `<action>` XML outputs. Replaced `matplotlib` with **TensorBoard** for real-time monitoring of SFT loss, RL loss, and format compliance rates.
*   **Curriculum Learning (SFT Warm-up)**: Solved the "cold-start deadlock" (where the model never produced a legal XML action in RL) by introducing an **SFT Warm-up Phase**. We generate a small dataset of optimal Ukeire trajectories and train the model via pure supervised fine-tuning (NLL) to establish formatting competence before triggering the RL strategy loop. The SFT warmup stage automatically saves a checkpoint (`sft_warmup_mahjong`) so this stage can be safely skipped in future RL runs.
*   **Hardware Constraints**: A 16GB GPU cannot run the `peft` preparation pipeline for Gemma models (even 2B variants) due to massive fp32 embedding casting. Phase 0 was verified using `Qwen2.5-0.5B-Instruct`. SFT phases utilize specific memory optimizations (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) and reduced batch sizes to prevent OOM.

### Phase 1: Heuristic Bootstrapping (GCP Cloud)
*   **Goal**: Enable a single LLM to master basic rules and tile efficiency.
*   **Environment**: Deployed to GCP Compute Engine (A100/H100) to safely allocate the massive VRAM required for Gemma-4 4B/9B models.
*   **Outcome**: A baseline model capable of stably deducing optimal discards using the Five Block Method, strictly adhering to the XML output format.

### Phase 2: Multi-Agent Self-Play Rollout
*   **Goal**: Emergence of defensive strategies and high-level board evaluation.
*   **Environment**: Deployment of 4 LLM instances (replicas of the same weights) running simultaneously.

### Phase 3: Cloud-Scale RL
*   **Optimization**: Tune the Advantage calculation parameters, PPO clipping constraints, and Rollout Batch Size to strike a balance between sampling costs and convergence speed.
