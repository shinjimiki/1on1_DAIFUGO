"""
トレーニングセットアップの確認テスト
"""
from train import DaifugoTrainer

print("Initializing trainer...")
trainer = DaifugoTrainer(seed=42)

print("Building PPO model...")
model = trainer.build_model(learning_rate=3e-4)
print(f"✓ PPO model built")
print(f"  Policy: {model.policy_class}")
print(f"  Learning rate: {model.learning_rate}")

print("\nTesting short training (100 steps)...")
trainer.train(total_timesteps=100)
print("✓ Training completed successfully")

print("\nEvaluating...")
stats = trainer.evaluate(num_episodes=10)
print(f"✓ Evaluation completed")
print(f"  Win rate: {stats['win_rate']:.2%}")
print(f"  Average reward: {stats['average_reward']:.2f}")

print("\n✓ All training components working!")
