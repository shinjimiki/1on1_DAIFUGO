"""
AlphaZero vs Random AI Demo
Demonstrate trained AlphaZero model
"""

import torch
import numpy as np
from datetime import datetime
import time

from alphazero_trainer import AlphaZeroTrainer
from alphazero_mcts import MCTS
from wrapper import DaifugoGymEnv


def play_alphazero_game(trainer: AlphaZeroTrainer, num_games: int = 5):
    """
    AlphaZero vs Random AI の対戦デモ

    Args:
        trainer: AlphaZero訓練クラス
        num_games: 対戦ゲーム数
    """
    print("="*70)
    print("🤖 AlphaZero vs Random AI 対戦デモ")
    print("="*70)
    print()

    wins = 0
    losses = 0
    total_steps = 0

    for game_num in range(num_games):
        print(f"🎮 Game {game_num + 1}/{num_games}")
        print("-"*40)

        env = DaifugoGymEnv(seed=int(time.time() * 1000000) % (2**31 - 1))
        env.player = 0  # AlphaZero is player 0
        obs, info = env.reset()

        mcts = MCTS(c_puct=1.0, max_simulations=400)  # 推論時は少ないシミュレーション
        step = 0
        max_steps = 1000

        while not env.state.done and step < max_steps:
            step += 1
            current_player = env.state.turn

            if current_player == 0:  # AlphaZeroのターン
                print(f"Step {step}: AlphaZero thinking...")

                # MCTSでアクション決定
                action_probs = mcts.search(env, trainer.network.policy_net, trainer.network.value_net)

                # 最も良いアクションを選択
                legal_mask = info["action_mask"]
                legal_probs = action_probs * legal_mask

                if legal_probs.sum() > 0:
                    action = np.argmax(legal_probs)
                else:
                    legal_ids = np.where(legal_mask)[0]
                    action = np.random.choice(legal_ids)

                print(f"  AlphaZero plays: action {action}")

            else:  # Random AIのターン
                print(f"Step {step}: Random AI thinking...")
                legal_mask = info["action_mask"]
                legal_ids = np.where(legal_mask)[0]
                action = np.random.choice(legal_ids)
                print(f"  Random AI plays: action {action}")

            # アクション実行
            obs, reward, done, _, info = env.step(action)

            if done:
                winner = "AlphaZero" if env.state.last_winner == 0 else "Random AI"
                print(f"\n🏆 Game {game_num + 1} finished! Winner: {winner} (steps: {step})")
                break

        total_steps += step

        if env.state.last_winner == 0:
            wins += 1
        else:
            losses += 1

        print()

    # 最終成績
    print("="*70)
    print("📊 Final Results")
    print("="*70)
    print(f"AlphaZero Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"Random AI Wins: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"Average Steps: {total_steps/num_games:.1f}")

    if wins > losses:
        print("\n🎉 AlphaZero is performing well!")
    elif wins == losses:
        print("\n🤝 It's a close match!")
    else:
        print("\n😅 AlphaZero needs more training!")


def main():
    """メイン関数"""
    print("Loading AlphaZero model...")

    try:
        trainer = AlphaZeroTrainer()
        trainer.load_model("models/alphazero_final.pth")
        print("✅ Model loaded successfully!")
        print()

        # 対戦デモ実行
        play_alphazero_game(trainer, num_games=5)

    except FileNotFoundError:
        print("❌ Model file not found: models/alphazero_final.pth")
        print("   Please train the model first:")
        print("   python alphazero_train.py")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


if __name__ == "__main__":
    main()