"""
interactive_play.py の簡単な動作テスト
（スクリプトのインポートと基本機能確認）
"""

print("✅ interactive_play.py のロードテスト中...\n")

try:
    from interactive_play import card_display, get_card_id, get_legal_action_choices
    from wrapper import DaifugoGymEnv
    
    print("✓ モジュールのインポート成功")
    
    # card_display テスト
    print("\n【card_display テスト】")
    test_cards = ["S3", "H10", "DJ", "C2", "JR", "JB"]
    for card in test_cards:
        print(f"  {card} → {card_display(card)}")
    
    # 環境初期化テスト
    print("\n【環境初期化テスト】")
    env = DaifugoGymEnv(seed=42)
    obs, info = env.reset()
    print(f"  観測ベクトル: {obs.shape}")
    print(f"  アクション空間: {env.action_space}")
    print(f"  合法手数: {int(info['action_mask'].sum())}")
    
    # 合法アクション取得テスト
    print("\n【合法アクション表示テスト】")
    choices = get_legal_action_choices(env)
    print(f"  合法手の数: {len(choices)}")
    for choice_text, action_id in choices[:5]:
        print(f"    {choice_text}")
    if len(choices) > 5:
        print(f"    ... ほか {len(choices) - 5} 個")
    
    print("\n✅ すべてのテストに成功しました！")
    print("\n次に 'python interactive_play.py' で対戦をお楽しみください！")
    
except Exception as e:
    print(f"\n❌ エラーが発生しました:\n{e}")
    import traceback
    traceback.print_exc()
