"""
自己対戦による強化学習のトレーニングループ。
PPO（Proximal Policy Optimization）を使用。
"""

from __future__ import annotations

import argparse
import numpy as np
from collections import deque
from datetime import datetime
import json
import os

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
    
    
    def save_model(self, path: str, save_info: bool = False, use_timestamp: bool = False) -> dict:
        """
        モデルを保存。
        
        Args:
            path: 保存先パス（use_timestamp=False の場合）または保存先ディレクトリ（use_timestamp=True の場合）
            save_info: True の場合、詳細情報ファイルも保存
            use_timestamp: True の場合、タイムスタンプ付きファイル名で保存（既存ファイルを上書きしない）
        
        Returns:
            保存情報の辞書
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        # タイムスタンプを用いたファイル名を生成
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if path.endswith(".zip"):
                base_path = path.replace(".zip", "")
            else:
                base_path = path
            save_path = f"{base_path}_{timestamp}.zip"
        else:
            save_path = path
        
        # ディレクトリが存在しない場合は作成
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        # 詳細情報を保存
        if save_info:
            now = datetime.now()
            info = {
                "timestamp": now.isoformat(),
                "datetime": now.strftime("%Y年%m月%d日 %H:%M:%S"),
                "model_path": save_path,
                "training_history": self.training_history,
                "env_config": {
                    "observation_space": str(self.env.observation_space),
                    "action_space": str(self.env.action_space),
                },
            }
            
            # JSON ファイルを保存
            info_path = save_path.replace(".zip", "_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"Model info saved to {info_path}")
            print(f"  Saved at: {info['datetime']}")
            
            return info
        
        return {"model_path": save_path}
    
    def load_model(self, path: str) -> None:
        """
        モデルを読み込み。
        
        Args:
            path: 読み込み元パス
        """
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
        
        # 情報ファイルが存在すれば読み込み
        info_path = path.replace(".zip", "_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                print(f"Model info loaded from {info_path}")
                print(f"  Saved at: {info.get('timestamp', 'N/A')}")
            except Exception as e:
                print(f"Warning: Could not load model info: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Daifugo PPO model")
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)"
    )
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-l", "--load",
        type=str,
        default=None,
        help="Path to pre-trained model to load and continue training"
    )
    parser.add_argument(
        "--save-info",
        action="store_true",
        help="Save detailed model info with timestamp"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Save with fixed name (default: use timestamp)"
    )
    
    args = parser.parse_args()
    
    # トレーニング開始
    trainer = DaifugoTrainer(seed=args.seed)
    
    # 既存モデルをロード
    if args.load:
        print(f"Loading model from {args.load}...")
        trainer.load_model(args.load)
        print("✓ Model loaded. Continuing training...")
    else:
        print("Building PPO model...")
        trainer.build_model(learning_rate=3e-4)
    
    print(f"Training for {args.timesteps} timesteps...")
    trainer.train(total_timesteps=args.timesteps)
    
    print(f"Evaluating over {args.episodes} episodes...")
    stats = trainer.evaluate(num_episodes=args.episodes)
    print(f"Evaluation stats: {stats}")
    
    # モデルを保存
    use_timestamp = not args.no_timestamp  # デフォルトはタイムスタンプを使用
    trainer.save_model(
        "models/daifugo_ppo.zip",
        save_info=args.save_info,
        use_timestamp=use_timestamp
    )
