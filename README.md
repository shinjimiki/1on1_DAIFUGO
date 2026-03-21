# 1on1_DAIFUGO

## 概要

**1対1 大富豪** の AI を強化学習（PPO）で訓練・対戦するシステムです。

- 🎮 完全ルール実装（革命、8切り、階段など）
- 🤖 PPO による強化学習
- 👥 人間 vs AI のインタラクティブ対戦
- 📊 Gymnasium 互換環境

## 仕様

- [RL specification (agreed)](./RL_SPEC.md)
- [GUI ユーザーガイド](./GUI_GUIDE.md)（グラフィカルモード用）

---

## 🚀 クイックスタート

### 1. セットアップ

```bash
# 仮想環境の作成（初回のみ）
python -m venv .venv

# 有効化
.venv\Scripts\activate

# 必要なパッケージをインストール
pip install gymnasium stable-baselines3 numpy
```

### 2. AI を訓練＆ゲームプレイ

```bash
# AI を訓練をして 5 ゲーム自動対戦
python play_game.py
```

**出力例:**

```
📚 モデルを訓練中（10,000ステップ）...
✅ 訓練完了！

🎯 5ゲームプレイ中...
成績: 5/5 勝利
勝率: 100.0%
💾 モデルを保存: models/daifugo_ppo.zip
```

### 3. 人間 vs AI で対戦（CLI版）

```bash
# インタラクティブゲーム（コマンドライン版）を開始
python interactive_play.py
```

**操作方法:**

1. モードを選択（訓練済みモデルまたはランダム）
2. あなたの手札が表示されます
3. 出すカードの番号を入力
4. AI の手札は秘密（枚数のみ表示）
5. 先に手札を 0 枚にした側が勝ち！

### 4. 人間 vs AI で対戦（GUI版）🆕

```bash
# グラフィカルインターフェース版を起動
python gui.py
```

**操作方法:**

- **クリック**: カードを選択（選択されたカードはハイライト）
- **Space キー**: 選択したカードで手を出す
- **P キー**: PASS（パス）して場を譲る
- **ESC キー**: カード選択を解除
- **← → キー**: 手札を横スクロール 🆕

**画面レイアウト:**

```
┌───────────────────────────────────────┐
│     AI の手札（裏）                    │
│    ? ? ? ? ? ? ? ?                     │
├───────────────────────────────────────┤
│ 場の状態 | 手札情報 | ターン情報        │
│ 合法手                                  │
├───────────────────────────────────────┤
│        あなたの手札（表）               │
│   ♠3  ♠4  ♥5  ♦6  ♣7 ...            │
└───────────────────────────────────────┘
```

---

## 📁 ファイル構成

| ファイル              | 説明                               |
| --------------------- | ---------------------------------- |
| `daifugo_env.py`      | コア環境実装（ゲームルール）       |
| `wrapper.py`          | Gymnasium 互換ラッパー             |
| `train.py`            | PPO トレーニング管理               |
| `play_game.py`        | AI 自動対戦デモ                    |
| `interactive_play.py` | 人間 vs AI インタラクティブ（CLI） |
| `gui.py`              | 🆕 人間 vs AI グラフィカルUI       |
| `RL_SPEC.md`          | ゲーム仕様書                       |

---

## 🎮 使用例

### パターン 1: AI を訓練して自動対戦

```python
from train import DaifugoTrainer

trainer = DaifugoTrainer(seed=42)
trainer.build_model(learning_rate=3e-4)
trainer.train(total_timesteps=100000)  # 大量訓練
stats = trainer.evaluate(num_episodes=100)
print(f"勝率: {stats['win_rate']:.1%}")
trainer.save_model("models/daifugo_ppo_v2.zip")
```

### パターン 2: 人間がプレイ

```bash
python interactive_play.py
# モード選択 → 1（訓練済みモデル）
# 手札の番号を入力して対戦
```

### パターン 3: ランダム AI と対戦

```bash
python interactive_play.py
# モード選択 → 2（ランダム）
# ランダムな AI と対戦
```

---

## 🔧 API リファレンス

### `Daifugo1v1Env`

```python
from daifugo_env import Daifugo1v1Env

env = Daifugo1v1Env(seed=42)
state = env.reset()

# 合法手を取得
legal_actions = env.legal_actions(state)

# 行動マスク（PPO用）
mask = env.action_mask(state)

# ステップ実行
next_state, reward, done, info = env.step(action_id)

# 状態をベクトルにエンコード
obs = env.encode_obs(state, player=0)
```

### `DaifugoTrainer`

```python
from train import DaifugoTrainer

trainer = DaifugoTrainer(seed=42)
trainer.build_model(learning_rate=3e-4)
trainer.train(total_timesteps=50000)
stats = trainer.evaluate(num_episodes=100)
trainer.save_model("path/to/model.zip")
trainer.load_model("path/to/model.zip")
```

---

## 📊 パフォーマンス

現在のモデル (`models/daifugo_ppo.zip`):

- **訓練ステップ**: 10,000
- **勝率**: 100% （5/5 ゲーム勝利）
- **平均報酬**: +1.0

---

## 🎯 今後の拡張予定

- [ ] 複数人対戦（3人以上）
- [ ] ゲーム UI（Web ブラウザ化）
- [ ] モデルの多様性評価
- [ ] 学習曲線の可視化
- [ ] セルフプレイ対戦トーナメント

---

## 📝 ライセンス

MIT License

---

## 🤝 貢献

改善提案やバグ報告は Issue でお願いします！
