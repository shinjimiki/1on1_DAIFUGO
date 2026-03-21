"""
Daifugo 環境の Gymnasium (Gym) 互換 Wrapper
PPO などの RL アルゴリズムで使用可能にする。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from daifugo_env import Daifugo1v1Env, State


class DaifugoGymEnv(gym.Env):
    """
    Daifugo1v1Env を Gymnasium 互換にした環境。
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.env = Daifugo1v1Env(seed=seed)
        self.player = 0  # 学習対象プレイヤー（0 か 1）
        self.opponent_policy = None  # 相手のポリシー（ランダムなど）
        
        # 行動空間: action_table のサイズ
        self.action_space = spaces.Discrete(len(self.env.action_table))
        
        # 観測空間: 146 次元
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(146,), dtype=np.float32
        )
        
        self.state: State | None = None
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        環境をリセット。
        
        Returns:
            obs: 観測ベクトル (146,)
            info: 情報辞書（action_mask を含む）
        """
        if seed is not None:
            self.env.rng.seed(seed)
        
        self.state = self.env.reset()
        
        # 初期手番が学習プレイヤーでない場合は、相手のアクションを実行
        while self.state.turn != self.player and not self.state.done:
            self._opponent_step()
        
        obs = self.env.encode_obs(self.state, self.player)
        mask = self.env.action_mask(self.state)
        return obs, {"action_mask": mask}
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        1ステップ実行。
        
        Args:
            action: 行動ID (action_table のインデックス)
        
        Returns:
            obs: 観測ベクトル
            reward: 報酬 (学習プレイヤーの視点)
            terminated: エピソード終了フラグ
            truncated: 時刻制限フラグ (未実装)
            info: 情報辞書
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        
        # action を int に変換（numpy array の場合も対応）
        action = int(action)
        
        # エピソード既に終了している場合は自動リセット
        if self.state.done:
            obs, info = self.reset()
            return obs, 0.0, False, False, info
        
        # 法的なアクションのマスクをチェック
        legal_actions = self.env.legal_actions(self.state)
        legal_ids = set(
            self.env.action_index[self.env._canon(a)] for a in legal_actions
        )
        
        # 非合法行動の場合は、合法行動の中からランダムに選択
        if action not in legal_ids:
            action = self.env.rng.choice(list(legal_ids))
        
        # 学習プレイヤーのアクション
        self.state, reward, done, info = self.env.step(action)
        
        # 相手のターンの間、相手がアクション
        while self.state.turn != self.player and not self.state.done:
            self._opponent_step()
        
        # 報酬を学習プレイヤーの視点に正規化
        if done:
            if self.player == 0:
                reward = reward  # 既に正しい (+1 or -1)
            else:
                reward = -reward  # プレイヤー1なので反転
        
        obs = self.env.encode_obs(self.state, self.player)
        mask = self.env.action_mask(self.state)
        
        return obs, float(reward), done, False, {"action_mask": mask}
    
    def _opponent_step(self) -> None:
        """相手のアクションを実行 (ランダムポリシー)"""
        if self.state is None or self.state.done:
            return
        
        legal_actions = self.env.legal_actions(self.state)
        legal_ids = [
            self.env.action_index[self.env._canon(a)] for a in legal_actions
        ]
        if not legal_ids:
            # 合法手がない場合はスキップ
            return
        
        action_id = self.env.rng.choice(legal_ids)
        self.state, _, done, _ = self.env.step(action_id)
    
    def render(self) -> None:
        """未実装"""
        pass
    
    def get_action_mask(self) -> np.ndarray:
        """現在の法的行動のマスクを返す"""
        if self.state is None:
            raise RuntimeError("Call reset() before get_action_mask().")
        return self.env.action_mask(self.state)
