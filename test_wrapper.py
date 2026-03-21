"""
Wrapper テスト
"""
from wrapper import DaifugoGymEnv

env = DaifugoGymEnv(seed=42)
obs, info = env.reset()

print('✓ Gymnasium wrapper initialized')
print(f'  Observation shape: {obs.shape}')
print(f'  Action space: {env.action_space}')
print(f'  Observation space: {env.observation_space}')

# Test step with legal action only
mask = info.get("action_mask", env.get_action_mask())
import numpy as np
legal_actions = np.where(mask == 1)[0]
action = legal_actions[0]

obs, reward, terminated, truncated, info = env.step(action)
print(f'✓ Step executed')
print(f'  Observation shape: {obs.shape}')
print(f'  Reward: {reward}')
print(f'  Terminated: {terminated}')
print(f'  Action mask shape: {info["action_mask"].shape}')
print(f'  Legal actions count: {int(info["action_mask"].sum())}')

print('\n✓ All tests passed!')

