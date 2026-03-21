# Copilot Instructions for `1on1_DAIFUGO`

## Project Overview

**1on1_DAIFUGO** - 1対1 大富豪 AI 強化学習システム

- 🎮 完全なルール実装（革命、8切り、階段、ジョーカーなど）
- 🤖 PPO（Proximal Policy Optimization）による強化学習
- 👥 人間 vs AI のインタラクティブ対戦システム
- 📊 Gymnasium 互換環境（強化学習標準環境）

---

## Build, test, and lint commands

### Setup (Initial)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install gymnasium stable-baselines3 numpy
```

### Run Commands

| Purpose | Command |
|---------|---------|
| **Train AI & Demo** | `python play_game.py` |
| **Play vs AI (Interactive)** | `python interactive_play.py` |
| **Test Environment** | `python test_wrapper.py` |
| **Test Training** | `python test_training.py` |
| **Test Interactive** | `python test_interactive.py` |

### Quick Test
```bash
python -c "from daifugo_env import Daifugo1v1Env; env = Daifugo1v1Env(); env.reset(); print('✓ Environment working')"
```

---

## High-level architecture

### Core Components

1. **`daifugo_env.py`** - Game Environment
   - `Daifugo1v1Env`: Main environment class
   - `State`: Game state dataclass
   - `Meld`: Card combination (single, group, straight)
   - Implements 大富豪 rules

2. **`wrapper.py`** - Gymnasium Adapter
   - `DaifugoGymEnv`: Gymnasium-compatible wrapper
   - Handles self-play (opponent AI uses random policy)
   - Converts actions to/from observation vectors

3. **`train.py`** - Training & Evaluation
   - `DaifugoTrainer`: Training management
   - PPO model configuration
   - Self-play training loop
   - Model save/load

4. **`interactive_play.py`** - Human vs AI
   - Interactive game loop
   - Human input handling
   - AI opponent integration
   - Hides opponent hand (only shows card count)

### Data Flow

```
Human Input
    ↓
interactive_play.py (parse & validate)
    ↓
DaifugoGymEnv (action selection)
    ↓
Daifugo1v1Env (game logic)
    ↓
State Update → Display
    ↓
AI Action (PPO model predict)
    ↓
[Repeat]
```

---

## Key conventions in this codebase

### Python Style
- Type hints required for function signatures
- Dataclasses for immutable data (`State`, `Meld`)
- Generator patterns for action enumeration
- NumPy arrays for observations (146-dim vectors)

### Game Constants
- RANKS = ["3", "4", ..., "A", "2"]  # 13 unique ranks
- SUITS = ["S", "H", "D", "C"]       # 4 suits
- JOKERS = ["JR", "JB"]              # 2 jokers
- Action table: 6,539 possible actions

### Environment Specs
- Observation space: 146-dimensional float32 vector
- Action space: Discrete(6539)
- Action mask: Boolean array of legal actions
- Reward: Terminal only (+1 for win, -1 for loss)

## Testing Strategy

### Unit Tests
- `test_wrapper.py`: Gymnasium wrapper functionality
- `test_training.py`: PPO model training
- `test_interactive.py`: Interactive mode parsing

### Quick Validation
```python
# Check environment
from daifugo_env import Daifugo1v1Env
env = Daifugo1v1Env(seed=42)
state = env.reset()
assert len(env.action_table) == 6539
assert len(env.legal_actions(state)) == 72

# Check wrapper
from wrapper import DaifugoGymEnv
env = DaifugoGymEnv()
obs, info = env.reset()
assert obs.shape == (146,)
```

---

## Important Notes for Future Sessions

1. **Model Location**: Trained models stored in `models/daifugo_ppo.zip`
2. **Environment State**: Fully deterministic with seed
3. **Hand Visibility**: Opponent hand always hidden in interactive mode
4. **Action Validation**: All actions must pass `action_mask` check
5. **Reward Structure**: Zero intermediate rewards, terminal-only (+1/-1)

## Documentation

- **[README.md](../README.md)**: Full project overview
- **[QUICKSTART.md](../QUICKSTART.md)**: User guide for interactive play
- **[RL_SPEC.md](../RL_SPEC.md)**: Detailed game rules & specifications

