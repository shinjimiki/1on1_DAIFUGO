# ジョーカー仕様書

## 概要

1v1 大富豪システムでのジョーカー（JR: 赤、JB: 黒）の扱いを定義する。

## ジョーカーの定義

- **JOKERS = ["JR", "JB"]** - 赤と黒のジョーカー
- **ランク**: ジョーカーは通常ランク（3-2など）を持たず、特殊ランク "JOKER" として扱われる
- **スート**: ジョーカーはスートを持たない

## 強さ比較

### 単体ジョーカーの力（比較対象が単体の場合）

- **通常時**: ジョーカー > 2（最強の数札）
- **革命時**: ジョーカー < 3（最弱の数札）
- ジョーカー同士の場合：同ランク扱い（JRとJBの区別なし）

### ジョーカーの強さ値

```python
rank_value("JOKER", revolution=False) = 14  # 最強
rank_value("JOKER", revolution=True) = -1   # 最弱
```

## 使用パターン

### 1. 単体出し

ジョーカーを1枚単独で出す。

```
例: JR 単体
例: JB 単体
```

**強さ**: 数札の最強の2と同じ扱い

---

### 2. グループアクション（同ランク）

#### パターン A: 数札 + ジョーカー

同じランクの数札とジョーカーを組み合わせて出す。

```
例: S3 + JR = (S3, JR) - 3のグループ（size=2）
例: S3 + H3 + JB = (S3, H3, JB) - 3のグループ（size=3）
例: S3 + H3 + C3 + JB = (S3, H3, C3, JB) - 3のグループ（size=4）
```

**特徴**:

- ジョーカーは数札の「ワイルドカード」として機能
- top_rank は **数札のランク** を使用（例：JBを含むグループでも top_rank="3"）
- ジョーカーの種類（JRか JBか）は区別しない

**強さ判定**: top_rank（数札）で比較

---

#### パターン B: ジョーカーのみ

ジョーカー複数枚だけのグループ。

```
例: JR + JB = (JB, JR) - ジョーカーペア（size=2）
```

**特徴**:

- size=2 のみ可能（ジョーカーは最大2枚）
- top_rank = **"JOKER"** として扱う
- ランク比較時は他のジョーカーのみのグループと同じ扱い

**強さ判定**: JOKER ランクで比較

---

### 3. ストレート（階段）

ジョーカーはワイルドカードとして置換可能。

```
例: S3-S4-S5 → 普通の階段
例: S3-JR-S5 → JRが S4 を置換（3-?-5 の階段）
例: S3-JR-JB → JR と JB が S4, S5 を置換（3-?-? の階段）
```

**詳細**: RL_SPEC.md 参照

---

## アクション生成ロジック

### グループアクション生成手順

1. **各ランク（3-2）について:**
   - size = 2, 3, 4 ごとに生成

2. **各 size について:**
   - joker_count = 0, 1, 2 ごとに処理
   - natural_count = size - joker_count

3. **ジョーカーがない場合（joker_count = 0）:**
   - 数札のみの組み合わせを生成

4. **ジョーカーある場合（joker_count >= 1）:**
   - 数札の全な部分集合と
   - ジョーカーの全な部分集合の **直積** を生成
   - 例: size=3, joker_count=1 の場合
     ```
     (数札3種から2枚を選ぶ) × (JRまたはJB)
     = 複数のパターン
     ```

5. **ジョーカーのみ（natural_count = 0）:**
   - top_rank = "JOKER" に設定
   - 例: (JB, JR) → rank="JOKER", size=2

### コード実装

```python
def _add_group_actions(self) -> None:
    for rank in RANKS:
        naturals = [f"{s}{rank}" for s in SUITS]
        for size in [2, 3, 4]:
            for joker_count in range(0, min(2, size) + 1):
                natural_count = size - joker_count

                if natural_count == 0:
                    # ジョーカーのみ
                    for joker_comb in combinations(JOKERS, joker_count):
                        cards = tuple(sorted(list(joker_comb)))
                        self._add_action(Meld("group", cards, size, "JOKER", None))
                else:
                    # 数札 + ジョーカー
                    for natural_comb in combinations(naturals, natural_count):
                        if joker_count == 0:
                            joker_combs = [()]
                        else:
                            joker_combs = list(combinations(JOKERS, joker_count))

                        for joker_comb in joker_combs:
                            cards = tuple(sorted(list(natural_comb) + list(joker_comb)))
                            self._add_action(Meld("group", cards, size, rank, None))
```

## 重要な実装詳細

### 1. ジョーカーの区別なし

- `JR` と `JB` は本質的に同じもの
- グループやストレートでは「2個のジョーカー」として扱う
- 表示・UI 上のみ JR/JB を区別可能

### 2. 正規化キー

アクション重複排除時は、ジョーカーの種類に関わらず同じアクションとして扱う：

```python
@staticmethod
def _canon(action: Meld | str) -> tuple:
    if action == PASS:
        return (PASS,)
    return (action.meld_type, tuple(sorted(action.cards)))
    # ソート済みカードリスト → JRとJBの順序は関係なし
```

### 3. アクションテーブルのサイズ

修正前: 6,539 アクション
修正後: 7,052+ アクション（ジョーカー組み合わせ追加）

---

## テスト用コード

```python
from daifugo_env import Daifugo1v1Env, JOKERS

# 両方のジョーカーを検証
for seed in range(100):
    env = Daifugo1v1Env(seed=seed)
    state = env.reset()

    hand = state.hands[0]
    if 'JR' in hand and 'JB' in hand:
        legal = env.legal_actions(state)

        # ジョーカー2枚のグループを検証
        both_joker = [a for a in legal
                     if a != 'PASS' and hasattr(a, 'cards')
                     and 'JR' in a.cards and 'JB' in a.cards]
        print(f"Actions with both JR+JB: {len(both_joker)}")
        break
```

---

## 参考

- **RL_SPEC.md**: 全体仕様
- **daifugo_env.py**: 実装
