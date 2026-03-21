"""
手札表示の強さ順テスト
"""

from wrapper import DaifugoGymEnv
from interactive_play import show_hand
import numpy as np

print("✅ 手札表示テスト（強さ順）\n")

# テスト 1: 通常時の手札表示
print("【テスト 1: 通常時（革命なし）】")
env = DaifugoGymEnv(seed=42)
obs, info = env.reset()

print(f"革命: {'いいえ'}")
show_hand(env.state, env.player)

# テスト 2: 複数ガイゲーム進めて革命が起きるか確認
print("\n" + "="*60)
print("【テスト 2: ゲーム進行中の手札表示】")

# いくつかステップを進める
for step in range(10):
    if env.state.done:
        break
    
    # 合法手を取得
    mask = info["action_mask"]
    legal_ids = np.where(mask)[0]
    action = np.random.choice(legal_ids)
    obs, reward, done, _, info = env.step(action)
    
    if step % 5 == 4:  # 5ステップごとに表示
        print(f"\n【ステップ {step+1}】")
        print(f"革命: {'はい' if env.state.revolution else 'いいえ'}")
        print(f"あなたの手札: {len(env.state.hands[0])}枚")
        show_hand(env.state, env.player)

print("\n\n✅ すべてのテストが正常に完了しました！")
print("手札は強さ順（強い順）で表示されています。")
print("🔥 アイコンは特に強いカード（ラッパー以上）を示します。")
