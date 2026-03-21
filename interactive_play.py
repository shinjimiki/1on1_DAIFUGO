"""
対人オンラインモード - 人間 vs AI
AI側の手札は秘密
"""

import time
from daifugo_env import Daifugo1v1Env, Meld, PASS, State, rank_of, rank_value, RANKS
from wrapper import DaifugoGymEnv
from train import DaifugoTrainer
import numpy as np


def card_display(card: str, suit_first: bool = True) -> str:
    """
    カードを見やすく表示
    Examples: S3, H10, DJ, JR
    """
    if card == "JR":
        return "🃏(赤)"
    elif card == "JB":
        return "🃏(黒)"
    else:
        suit_map = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
        suit = suit_map.get(card[0], card[0])
        rank = card[1:]
        return f"{suit}{rank}"


def get_card_id(display: str) -> str | None:
    """表示形式をカードIDに変換"""
    display = display.strip().upper()
    
    # スーツ
    suit_map = {"S": "S", "♠": "S", "H": "H", "♥": "H", "D": "D", "♦": "D", "C": "C", "♣": "C"}
    
    # ジョーカー
    if "赤" in display or display == "JR":
        return "JR"
    if "黒" in display or display == "JB":
        return "JB"
    
    # 通常カード (例: S3, ♠10)
    if len(display) >= 2:
        suit_char = display[0]
        rank_part = display[1:]
        
        suit = suit_map.get(suit_char)
        if suit and rank_part:
            return f"{suit}{rank_part}"
    
    return None


def display_game_state(env, player: int, verbose: bool = True):
    """ゲーム状態を表示"""
    state = env.state
    
    if not verbose:
        return
    
    print("\n" + "="*70)
    print(f"【ターン情報】")
    print(f"  現在のターン: {'あなた (Player 0)' if state.turn == 0 else 'AI (Player 1)'}")
    print(f"  革命: {'🔄 発動中！' if state.revolution else '通常'}")
    
    print(f"\n【手札情報】")
    print(f"  あなたの手札: {len(state.hands[0])}枚")
    print(f"  AI の手札: {len(state.hands[1])}枚（秘密）")
    
    print(f"\n【場の状態】")
    if state.table is None:
        print(f"  場: 🟤 クリア（新しい場を開始します）")
    else:
        table_type_map = {"single": "シングル", "group": "グループ", "straight": "階段"}
        table_type = table_type_map[state.table.meld_type]
        cards_str = " ".join([card_display(c) for c in sorted(state.table.cards)])
        print(f"  場: 【{table_type}】({state.table.size}枚) {cards_str}")
        print(f"      最高ランク: {state.table.top_rank}")


def show_hand(state: State, player: int):
    """人間の手札を表示（強さ順）"""
    hand = list(state.hands[player])
    
    # 強さでソート（強い順）
    # 革命フラグを考慮して値を計算
    def card_strength(card: str) -> tuple:
        rank = rank_of(card)
        value = rank_value(rank, state.revolution)
        # (強さ値（降順）, スート, ランク) でソート
        return (-value, card[0] if len(card) > 1 else "", rank)
    
    hand_sorted = sorted(hand, key=card_strength)
    
    print("\n【あなたの手札】")
    if not hand:
        print("  手札がありません！（勝利しました！）")
        return
    
    # 現在の状態を表示
    if state.revolution:
        print("  🔄 革命中のため、ランク順が反転しています")
    
    for i, card in enumerate(hand_sorted, 1):
        strength_indicator = "🔥" if rank_value(rank_of(card), state.revolution) >= 10 else "  "
        print(f"  {i:2d}. {strength_indicator} {card_display(card)}")



def get_legal_action_choices(env) -> list:
    """
    人間が選択できる合法アクションをリスト化（強さ順）
    戻り値: [(表示文字列, アクションID), ...]
    """
    state = env.state
    legal_actions = env.env.legal_actions(state)
    legal_ids = [env.env.action_index[env.env._canon(a)] for a in legal_actions]
    
    choices = []
    
    for action_id in legal_ids:
        action = env.env.action_table[action_id]
        if action == PASS:
            choices.append((f"PASS（パス）", action_id, -999))  # 最後に表示するため低い値を付ける
        else:
            cards_str = " ".join([card_display(c) for c in sorted(action.cards)])
            meld_type_map = {"single": "単", "group": "グ", "straight": "直"}
            meld_type = meld_type_map[action.meld_type]
            # 強さ値でソートするために top_rank の強さを取得
            strength = rank_value(action.top_rank, state.revolution)
            choices.append(
                (f"[{meld_type}] {cards_str}", action_id, strength)
            )
    
    # 強さ順でソート（PASS は最後に）
    choices_sorted = sorted(choices, key=lambda x: -x[2])
    
    # インデックスを付け直す
    result = []
    for idx, (text, action_id, _) in enumerate(choices_sorted, 1):
        result.append((f"{idx}. {text}", action_id))
    
    return result


def play_interactive_game(model=None):
    """
    人間が対戦するインタラクティブゲーム
    
    Args:
        model: 訓練済みモデル（PPO）
    """
    # 環境初期化
    env = DaifugoGymEnv(seed=int(time.time() * 1000000) % (2**31 - 1))
    env.player = 0  # 人間がプレイヤー0
    obs, info = env.reset()
    
    print("\n" + "="*70)
    print("🎮 1v1 大富豪 - 人間 vs AI")
    print("="*70)
    print("\n【ルール説明】")
    print("  - 手札をすべて出せば勝ちです")
    print("  - 場に出ている役より強い役を出してください")
    print("  - 出せない場合は PASS してください")
    print("  - AI の手札は秘密です\n")
    
    step = 0
    max_steps = 1000
    
    while step < max_steps:
        step += 1
        
        if env.state.done:
            print("\n" + "="*70)
            print("【ゲーム終了】")
            if env.state.last_winner == 0:
                print("🎉 あなたの勝利です！おめでとうございます！")
            else:
                print("😢 AI の勝利です。次回スのチャレンジをお祈りします！")
            print("="*70 + "\n")
            break
        
        # 現在のターンプレイヤー
        turn = env.state.turn
        
        # ゲーム状態表示
        display_game_state(env, turn)
        
        if turn == env.player:
            # 人間のターン
            print("\n【あなたのターン】")
            show_hand(env.state, env.player)
            
            # 合法アクション表示
            choices = get_legal_action_choices(env)
            print("\n【出すカード選択】")
            for choice_text, _ in choices:
                print(f"  {choice_text}")
            
            # ユーザー入力
            while True:
                try:
                    choice = input("\n番号を入力してください > ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(choices):
                        _, action_id = choices[choice_idx]
                        break
                    else:
                        print(f"❌ 1～{len(choices)} の番号を入力してください")
                except ValueError:
                    print("❌ 数字を入力してください")
        else:
            # AI のターン
            print("\n【AI のターン】")
            if model is not None:
                action_id, _ = model.predict(obs, deterministic=True)
                action_id = int(action_id)
            else:
                # ランダムプレイ
                mask = info["action_mask"]
                legal_ids = np.where(mask)[0]
                action_id = np.random.choice(legal_ids)
            
            # AI のアクション表示
            action = env.env.action_table[int(action_id)]
            if action == PASS:
                print("  🤖 AI: PASS")
            else:
                cards_str = " ".join([card_display(c) for c in sorted(action.cards)])
                meld_type_map = {"single": "単", "group": "グ", "straight": "直"}
                meld_type = meld_type_map[action.meld_type]
                print(f"  🤖 AI: [{meld_type}] {cards_str}")
            
            input("\nEnter キーで続行...")
        
        # ステップ実行
        obs, reward, done, _, info = env.step(action_id)
        print()


def main():
    print("\n" + "="*70)
    print("🎮 大富豪 AI との対戦システム")
    print("="*70)
    
    print("\n【モード選択】")
    print("  1. 訓練済みモデルと対戦")
    print("  2. AI（ランダム）と対戦")
    
    while True:
        mode = input("\nモードを選択してください (1 or 2) > ").strip()
        if mode in ["1", "2"]:
            break
        print("❌ 1 または 2 を入力してください")
    
    if mode == "1":
        print("\n📚 訓練済みモデルをロード中...")
        try:
            trainer = DaifugoTrainer()
            trainer.load_model("models/daifugo_ppo.zip")
            model = trainer.model
            print("✅ モデルをロード完了！\n")
        except FileNotFoundError:
            print("❌ モデルが見つかりません (models/daifugo_ppo.zip)")
            print("   まず play_game.py を実行してモデルを訓練してください\n")
            return
    else:
        print("\n🎲 ランダムモードで対戦します\n")
        model = None
    
    # ゲーム開始
    while True:
        play_interactive_game(model=model)
        
        again = input("もう一度プレイしますか？ (y/n) > ").strip().lower()
        if again != "y":
            print("\n👋 プレイありがとうございました！\n")
            break


if __name__ == "__main__":
    main()
