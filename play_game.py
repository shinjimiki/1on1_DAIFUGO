"""
訓練済みモデルでゲームをプレイ＆可視化
"""

import time
from daifugo_env import Daifugo1v1Env, rank_of, RANKS
from wrapper import DaifugoGymEnv
from train import DaifugoTrainer
import numpy as np


def card_display(card: str) -> str:
    """カードを見やすく表示"""
    if card == "JR":
        return "🃏R"
    elif card == "JB":
        return "🃏B"
    else:
        return f"{card[0]}{card[1:]}"


def display_state(env: DaifugoGymEnv, turn: int, verbose: bool = False):
    """ゲーム状態を表示"""
    state = env.state
    
    if verbose:
        print("\n" + "="*60)
        print(f"📍 ターン: {'私' if turn == env.player else '相手'} (Player {turn})")
        print(f"   革命: {'🔄 発動中' if state.revolution else '通常'}")
        print(f"   手札枚数 - 私: {len(state.hands[env.player])}枚, 相手: {len(state.hands[1-env.player])}枚")
        
        if state.table is None:
            print("   📋 場: クリア")
        else:
            table_type = {"single": "シングル", "group": "グループ", "straight": "階段"}[state.table.meld_type]
            cards_str = " ".join([card_display(c) for c in sorted(state.table.cards)])
            print(f"   📋 場: {table_type} ({state.table.size}枚) {cards_str} 【{state.table.top_rank}】")


def play_one_game(model=None, verbose: bool = True):
    """
    1ゲームプレイ
    
    Args:
        model: 訓練済みモデル（Noneの場合はランダムプレイ）
        verbose: 詳細出力
    """
    env = DaifugoGymEnv(seed=int(time.time() * 1000000) % (2**31 - 1))
    env.player = 0  # プレイヤー0を学習対象
    obs, info = env.reset()
    
    episode_steps = 0
    max_steps = 1000  # 無限ループ防止
    
    while True:
        episode_steps += 1
        if episode_steps > max_steps:
            print("⚠️  最大ステップ数に達しました")
            break
        
        turn = env.state.turn
        
        if model is not None:
            # モデルで行動選択
            if turn == env.player:
                action, _ = model.predict(obs, deterministic=True)
                action_source = "🤖"
            else:
                # 相手はランダム
                mask = info["action_mask"]
                legal_ids = np.where(mask)[0]
                if len(legal_ids) == 0:
                    print(f"ERROR: No legal actions available for Player {turn}")
                    print(f"  Mask sum: {np.sum(mask)}, Mask length: {len(mask)}")
                    print(f"  Game done: {env.state.done}")
                    break
                action = np.random.choice(legal_ids)
                action_source = "🎲"
        else:
            # ランダムプレイ
            mask = info["action_mask"]
            legal_ids = np.where(mask)[0]
            if len(legal_ids) == 0:
                print(f"ERROR: No legal actions available for Player {turn}")
                print(f"  Mask sum: {np.sum(mask)}, Mask length: {len(mask)}")
                print(f"  Game done: {env.state.done}")
                break
            action = np.random.choice(legal_ids)
            action_source = "🎲"
        
        # アクション実行前の状態表示
        if verbose:
            display_state(env, turn)
            
            # 出されたアクション表示
            action_obj = env.env.action_table[int(action)]
            if action_obj == "PASS":
                print(f"   {action_source} → PASS")
            else:
                cards_str = " ".join([card_display(c) for c in sorted(action_obj.cards)])
                meld_type = {"single": "単", "group": "グ", "straight": "直"}[action_obj.meld_type]
                print(f"   {action_source} → [{meld_type}] {cards_str}")
        
        # ステップ実行
        obs, reward, done, _, info = env.step(action)
        
        if done:
            if verbose:
                display_state(env, 1 - env.state.turn, verbose=True)
                winner = env.state.last_winner
                winner_name = "私（プレイヤー0）" if winner == 0 else "相手（プレイヤー1）"
                print(f"\n🎉 ゲーム終了! 勝者: {winner_name}")
                print(f"   総ステップ数: {episode_steps}")
            
            return env.state.last_winner == env.player


def main():
    print("🎮 1v1 大富豪 - AI対戦デモ\n")
    
    # モデルの訓練
    print("📚 モデルを訓練中（10,000ステップ）...\n")
    trainer = DaifugoTrainer(seed=42)
    trainer.build_model(learning_rate=3e-4)
    trainer.train(total_timesteps=10000)
    
    print("\n✅ 訓練完了！\n")
    
    # 複数ゲームをプレイ
    num_games = 5
    wins = 0
    
    print(f"🎯 {num_games}ゲームプレイ中...\n")
    
    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"🎮 ゲーム {game_num + 1}/{num_games}")
        print(f"{'='*60}")
        
        win = play_one_game(model=trainer.model, verbose=True)
        if win:
            wins += 1
    
    # 統計表示
    print(f"\n{'='*60}")
    print(f"📊 最終結果")
    print(f"{'='*60}")
    print(f"成績: {wins}/{num_games} 勝利")
    print(f"勝率: {wins/num_games*100:.1f}%")
    
    # モデル保存
    import os
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/daifugo_ppo.zip")
    print(f"\n💾 モデルを保存: models/daifugo_ppo.zip")


if __name__ == "__main__":
    main()
