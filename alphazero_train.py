"""
AlphaZero Training Script for Daifugo
Command-line interface for training AlphaZero models
"""

import argparse
from datetime import datetime
import torch

from alphazero_trainer import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero model for Daifugo")
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=50,
        help="Number of training iterations (default: 50)"
    )
    parser.add_argument(
        "-g", "--games",
        type=int,
        default=50,
        help="Games per iteration (default: 50)"
    )
    parser.add_argument(
        "-l", "--load",
        type=str,
        default=None,
        help="Path to pre-trained model to continue training"
    )
    parser.add_argument(
        "-s", "--save",
        type=str,
        default="models/alphazero_final.pth",
        help="Path to save the trained model (default: models/alphazero_final.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)"
    )

    args = parser.parse_args()

    print("="*70)
    print("🎯 AlphaZero Training for Daifugo")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"Total games: {args.iterations * args.games}")
    print()

    # 訓練クラス初期化
    trainer = AlphaZeroTrainer(device=args.device)

    # 既存モデルをロード
    if args.load:
        print(f"Loading model from {args.load}...")
        trainer.load_model(args.load)
        print("✅ Model loaded. Continuing training...")
    else:
        print("Starting training from scratch...")

    print()

    # 訓練実行
    try:
        trainer.train(
            num_iterations=args.iterations,
            games_per_iteration=args.games
        )

        # 最終モデル保存
        print(f"\n💾 Saving final model to {args.save}...")
        trainer.save_model(args.save)

        print("\n✅ Training completed successfully!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        # 中断時もモデルを保存
        interrupt_save_path = f"models/alphazero_interrupted_{int(datetime.now().timestamp())}.pth"
        print(f"Saving interrupted model to {interrupt_save_path}...")
        trainer.save_model(interrupt_save_path)

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()