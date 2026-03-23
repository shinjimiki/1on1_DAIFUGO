"""
AlphaZero Trainer for Daifugo
Self-play training with MCTS and neural networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import random
from datetime import datetime
import json
import os

from alphazero_network import AlphaZeroNet
from alphazero_mcts import MCTS
from wrapper import DaifugoGymEnv


class SelfPlayData(Dataset):
    """自己対戦データのデータセット"""

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, policy_target, value_target = self.data[idx]
        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(policy_target),
            torch.FloatTensor([value_target])
        )


class AlphaZeroTrainer:
    """AlphaZero訓練クラス"""

    def __init__(self, input_size: int = 146, action_size: int = 7052, device: str = "auto"):
        self.input_size = input_size
        self.action_size = action_size

        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # ネットワーク初期化
        self.network = AlphaZeroNet(input_size, action_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # MCTS設定
        self.mcts = MCTS(c_puct=1.0, max_simulations=400)

        # 訓練データ
        self.replay_buffer = deque(maxlen=100000)
        self.training_history = []

    def self_play_game(self, env: DaifugoGymEnv) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        1ゲームの自己対戦を実行し、訓練データを生成

        Returns:
            game_data: [(observation, policy_target, value_target), ...]
        """
        game_data = []
        obs, info = env.reset()

        while not env.state.done:
            # MCTSでアクション確率を計算
            action_probs = self.mcts.search(env, self.network.policy_net, self.network.value_net)

            # アクションを選択（確率的）
            legal_mask = info["action_mask"]
            legal_probs = action_probs * legal_mask
            if legal_probs.sum() > 0:
                legal_probs = legal_probs / legal_probs.sum()
                action = np.random.choice(len(legal_probs), p=legal_probs)
            else:
                # フォールバック
                legal_ids = np.where(legal_mask)[0]
                action = np.random.choice(legal_ids)

            # データを保存（現在の観測とMCTS確率）
            game_data.append((obs.copy(), action_probs.copy(), 0.0))  # valueは後で設定

            # アクション実行
            obs, reward, done, _, info = env.step(action)

        # ゲーム終了時の価値をバックアップ
        final_value = 1.0 if env.state.last_winner == 0 else -1.0
        for i in range(len(game_data)):
            # プレイヤー0の視点から価値を設定
            player_at_move = (len(game_data) - 1 - i) % 2
            value = final_value if player_at_move == 0 else -final_value
            obs, policy, _ = game_data[i]
            game_data[i] = (obs, policy, value)

        return game_data

    def train_network(self, batch_size: int = 64, epochs: int = 10):
        """ネットワークを訓練"""
        if len(self.replay_buffer) < batch_size:
            print(f"Not enough data for training: {len(self.replay_buffer)} < {batch_size}")
            return

        # データセット作成
        sample_data = random.sample(self.replay_buffer, min(len(self.replay_buffer), 10000))
        dataset = SelfPlayData(sample_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for obs_batch, policy_batch, value_batch in dataloader:
                obs_batch = obs_batch.to(self.device)
                policy_batch = policy_batch.to(self.device)
                value_batch = value_batch.to(self.device)

                self.optimizer.zero_grad()

                # 順伝播
                pred_policy, pred_value = self.network(obs_batch)

                # 損失計算
                policy_loss = nn.CrossEntropyLoss()(pred_policy, policy_batch)
                value_loss = nn.MSELoss()(pred_value.squeeze(), value_batch.squeeze())
                loss = policy_loss + value_loss

                # 逆伝播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss

        avg_loss = total_loss / epochs
        self.scheduler.step()

        print(".4f")
        return avg_loss

    def train(self, num_iterations: int = 100, games_per_iteration: int = 100):
        """
        AlphaZero訓練を実行

        Args:
            num_iterations: イテレーション数
            games_per_iteration: 1イテレーションあたりの自己対戦ゲーム数
        """
        print(f"Starting AlphaZero training for {num_iterations} iterations...")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Total games: {num_iterations * games_per_iteration}")

        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # 自己対戦でデータ生成
            print("Generating self-play data...")
            iteration_data = []

            for game in range(games_per_iteration):
                env = DaifugoGymEnv(seed=int(datetime.now().timestamp() * 1000000) % (2**31 - 1))
                game_data = self.self_play_game(env)
                iteration_data.extend(game_data)

                if (game + 1) % 20 == 0:
                    print(f"  Completed {game + 1}/{games_per_iteration} games")

            # リプレイバッファに追加
            self.replay_buffer.extend(iteration_data)
            print(f"Added {len(iteration_data)} data points to replay buffer")
            print(f"Total buffer size: {len(self.replay_buffer)}")

            # ネットワーク訓練
            print("Training network...")
            avg_loss = self.train_network()

            # 進捗保存
            self.training_history.append({
                "iteration": iteration + 1,
                "games_played": (iteration + 1) * games_per_iteration,
                "buffer_size": len(self.replay_buffer),
                "avg_loss": avg_loss,
                "timestamp": datetime.now().isoformat()
            })

            # 定期的にモデル保存
            if (iteration + 1) % 10 == 0:
                self.save_model(f"models/alphazero_iter_{iteration + 1}.pth")

        print("\nTraining completed!")

    def save_model(self, path: str):
        """モデルを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'replay_buffer_size': len(self.replay_buffer)
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {path}")
        print(f"Training history: {len(self.training_history)} iterations")

    def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """推論"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        policy_probs, value = self.network.predict(obs_tensor)
        return policy_probs.squeeze(0).cpu().numpy(), value.item()