"""
自己対戦による強化学習のトレーニングループ。
PPO（Proximal Policy Optimization）を使用。
"""

from __future__ import annotations

import numpy as np
from collections import deque

# stable-baselines3 のインポート（後でインストール）
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.policies import MlpPolicy
except ImportError:
    print("Warning: stable_baselines3 not installed. Install with: pip install stable-baselines3")

from wrapper import DaifugoGymEnv


class SelfPlayCallback(BaseCallback):
    """
    自己対戦イテレーションごとのコールバック。
    定期的にモデルをコピーして相手ポリシーとする。
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.episode_rewards = deque(maxlen=100)
        self.model_updates = 0
    
    def _on_step(self) -> bool:
        """各ステップ後に呼び出される"""
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.episode_rewards.append(mean_reward)
            
            if len(self.episode_rewards) == 100:
                avg_reward = np.mean(self.episode_rewards)
                if self.verbose > 0:
                    print(f"Mean reward (last 100 ep): {avg_reward:.2f}")
        
        return True


class DaifugoTrainer:
    """
    1v1 大富豪の自己対戦トレーニング。
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.env = DaifugoGymEnv(seed=seed)
        self.model = None
        self.training_history = []
    
    def build_model(
        self,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
    ) -> PPO:
        """
        PPO エージェントを構築。
        
        Args:
            policy: ポリシーネットワークの種類
            learning_rate: 学習率
            gamma: 割引率
            gae_lambda: GAE ラムダ
            clip_range: PPO クリップ範囲
        
        Returns:
            PPO モデル
        """
        self.model = PPO(
            policy,
            self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            n_epochs=10,
            batch_size=64,
            n_steps=2048,
            verbose=1,
            seed=self.seed,
        )
        return self.model
    
    def train(self, total_timesteps: int = 100000) -> None:
        """
        トレーニング実行。
        
        Args:
            total_timesteps: 総ステップ数
        """
        if self.model is None:
            self.build_model()
        
        callback = SelfPlayCallback(verbose=1)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )
        self.training_history.append({
            "timesteps": total_timesteps,
            "rewards": list(callback.episode_rewards),
        })
    
    def evaluate(self, num_episodes: int = 100) -> dict:
        """
        トレーニング済みモデルを評価。
        
        Args:
            num_episodes: 評価エピソード数
        
        Returns:
            評価統計
        """
        if self.model is None:
            raise RuntimeError("Train model first.")
        
        wins = 0
        losses = 0
        rewards = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                ep_reward += reward
            
            rewards.append(ep_reward)
            if ep_reward > 0:
                wins += 1
            elif ep_reward < 0:
                losses += 1
        
        win_rate = wins / num_episodes
        avg_reward = np.mean(rewards)
        
        return {
            "num_episodes": num_episodes,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "average_reward": avg_reward,
        }
    
    def save_model(self, path: str) -> None:
        """モデルを保存"""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """モデルを読み込み"""
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # トレーニング例
    trainer = DaifugoTrainer(seed=42)
    
    print("Building PPO model...")
    trainer.build_model(learning_rate=3e-4)
    
    print("Training...")
    trainer.train(total_timesteps=50000)
    
    print("Evaluating...")
    stats = trainer.evaluate(num_episodes=100)
    print(f"Evaluation stats: {stats}")
    
    trainer.save_model("models/daifugo_ppo.zip")
