from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Literal

import numpy as np


Suit = Literal["S", "H", "D", "C"]
MeldType = Literal["single", "group", "straight"]

RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUITS: list[Suit] = ["S", "H", "D", "C"]
JOKERS = ["JR", "JB"]
PASS = "PASS"


def make_deck() -> list[str]:
    deck = [f"{s}{r}" for s in SUITS for r in RANKS] + JOKERS
    return deck


def is_joker(card_id: str) -> bool:
    return card_id in JOKERS


def rank_of(card_id: str) -> str:
    if is_joker(card_id):
        return "JOKER"
    return card_id[1:]


def suit_of(card_id: str) -> Suit | None:
    if is_joker(card_id):
        return None
    return card_id[0]  # type: ignore[return-value]


def rank_value(rank: str, revolution: bool) -> int:
    if rank == "JOKER":
        return -1 if revolution else len(RANKS) + 1
    base = RANKS.index(rank)
    return -base if revolution else base


@dataclass(frozen=True)
class Meld:
    meld_type: MeldType
    cards: tuple[str, ...]
    size: int
    top_rank: str
    suit: Suit | None = None


@dataclass
class State:
    hands: list[set[str]]
    table: Meld | None
    turn: int
    revolution: bool
    done: bool
    last_winner: int


class Daifugo1v1Env:
    """
    1v1大富豪の最小ひな形環境。
    仕様詳細は RL_SPEC.md を参照。
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.action_table: list[Meld | str] = []
        self.action_index: dict[tuple, int] = {}
        self.state: State | None = None
        self._build_action_table()

    def reset(self) -> State:
        deck = make_deck()
        self.rng.shuffle(deck)
        p0 = set(deck[:27])
        p1 = set(deck[27:])
        self.state = State(
            hands=[p0, p1],
            table=None,
            turn=0,
            revolution=False,
            done=False,
            last_winner=0,
        )
        return self.state

    def legal_actions(self, state: State) -> list[Meld | str]:
        """
        現在の手札と場の状態から、法的な行動をすべて列挙する。
        
        場がない場合: すべての有効な Meld を返す
        場がある場合: 場をbeat可能な Meld と PASS を返す
        """
        hand = state.hands[state.turn]
        hand_list = sorted(hand)
        
        # 手札から生成可能なすべての Meld を取得
        candidates = self._generate_melds(hand_list)
        
        if state.table is None:
            # 場がない場合: すべての candidates を返す
            return candidates
        
        # 場がある場合: beat 可能な candidates をフィルタリング
        out: list[Meld | str] = []
        table = state.table
        
        for meld in candidates:
            if not self._can_beat(meld, table, state.revolution):
                continue
            out.append(meld)
        
        out.append(PASS)
        return out
    
    def _generate_melds(self, hand_list: list[str]) -> list[Meld]:
        """
        手札から生成可能なすべての Meld を列挙する。
        (単体、グループ、階段)
        """
        melds: list[Meld] = []
        hand_set = set(hand_list)
        card_ids = self._all_54_cards()
        
        # action_table 内で、手札に含まれるカードのみで構成される Meld をフィルタリング
        for action in self.action_table:
            if action == PASS:
                continue
            meld = action
            if all(c in hand_set for c in meld.cards):
                melds.append(meld)
        
        return melds
    
    def _can_beat(self, candidate: Meld, table: Meld, revolution: bool) -> bool:
        """
        candidate が table をbeat可能か判定する。
        - 同じ役タイプ
        - 同じサイズ（階段は同じ長さ）
        - top_rank の強さが上回る
        """
        if candidate.meld_type != table.meld_type:
            return False
        if candidate.size != table.size:
            return False
        
        # 強さ比較
        c_value = rank_value(candidate.top_rank, revolution)
        t_value = rank_value(table.top_rank, revolution)
        return c_value > t_value

    def step(self, action_id: int) -> tuple[State, float, bool, dict]:
        """
        1ステップ実行。アクションを実行し、状態、報酬、終了フラグ、情報を返す。
        
        報酬:
        - 自分が勝利: +1
        - 自分が敗北: -1
        - その他: 0
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.done:
            raise RuntimeError("Episode already done. Call reset().")

        action = self.action_table[action_id]
        legal = self.legal_actions(self.state)
        legal_keys = {self._canon(a) for a in legal}
        if self._canon(action) not in legal_keys:
            raise ValueError("Illegal action for current state.")

        p = self.state.turn
        
        if action == PASS:
            # PASS: 場流れ、対手に手番を渡す
            self.state.table = None
            self.state.turn = 1 - p
            return self.state, 0.0, False, {"event": "pass_flow"}

        meld = action
        assert isinstance(meld, Meld)
        
        # カードを手札から削除
        for c in meld.cards:
            self.state.hands[p].remove(c)

        # 4枚出し（同ランク4枚）の場合は革命トグル
        if meld.meld_type == "group" and meld.size == 4:
            self.state.revolution = not self.state.revolution

        # 8 を含むかチェック
        contains_8 = any(rank_of(c) == "8" for c in meld.cards)
        
        if contains_8:
            # 8切り: 場流れ、同じプレイヤーが次の場を開始
            self.state.table = None
            self.state.turn = p
            self.state.last_winner = p
        else:
            # 通常: テーブルセット、対手に手番を渡す
            self.state.table = meld
            self.state.turn = 1 - p
            self.state.last_winner = p

        # 手札が 0 枚になったら終了（勝利）
        if len(self.state.hands[p]) == 0:
            self.state.done = True
            # p が勝者
            if p == 0:
                reward = 1.0
            else:
                reward = -1.0
            return self.state, reward, True, {"winner": p}

        return self.state, 0.0, False, {}

    def encode_obs(self, state: State, player: int) -> np.ndarray:
        """
        状態を 132 次元のベクトルにエンコードする。
        
        構成:
        - [0:54]: プレイヤーの手札（54bit = 52枚 + Joker2枚）
        - [54:108]: 相手の手札（54bit）
        - [108:112]: 場の役タイプ（one-hot: none/single/group/straight）
        - [112:126]: 場の枚数（one-hot: 0..13）
        - [126:140]: 場の基準ランク（one-hot: 3..2）
        - [140:144]: 場のスート（one-hot: S/H/D/C）
        - [144]: 革命フラグ
        - [145]: 手番フラグ（自プレイヤーなら1）
        
        合計: 146次元（仕様では132とあるが、確認が必要）
        """
        obs = np.zeros(146, dtype=np.float32)
        card_ids = self._all_54_cards()
        card_to_idx = {c: i for i, c in enumerate(card_ids)}

        # [0:54] 自分の手札
        for c in state.hands[player]:
            obs[card_to_idx[c]] = 1.0
        
        # [54:108] 相手の手札
        for c in state.hands[1 - player]:
            obs[54 + card_to_idx[c]] = 1.0

        # [108:144] 場情報
        offset = 108
        if state.table is None:
            # [108]: none（場がない）
            obs[offset + 0] = 1.0
        else:
            # [108:112] 役タイプ one-hot
            meld_type_map = {"single": 1, "group": 2, "straight": 3}
            t = meld_type_map[state.table.meld_type]
            obs[offset + t] = 1.0
            
            # [112:126] 枚数 one-hot (size が 1..13 の範囲)
            obs[offset + 4 + state.table.size] = 1.0
            
            # [126:140] 基準ランク one-hot (RANKS に基づいて)
            if state.table.top_rank in RANKS:
                rank_idx = RANKS.index(state.table.top_rank)
                obs[offset + 18 + rank_idx] = 1.0
            
            # [140:144] スート one-hot (straight のみで使用)
            if state.table.suit is not None:
                suit_idx = SUITS.index(state.table.suit)
                obs[offset + 32 + suit_idx] = 1.0

        # [144] 革命フラグ
        obs[144] = 1.0 if state.revolution else 0.0
        
        # [145] 手番フラグ
        obs[145] = 1.0 if state.turn == player else 0.0
        
        return obs

    def action_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(len(self.action_table), dtype=np.float32)
        for action in self.legal_actions(state):
            mask[self.action_index[self._canon(action)]] = 1.0
        return mask

    def _build_action_table(self) -> None:
        self._add_action(PASS)
        self._add_single_actions()
        self._add_group_actions()
        self._add_straight_actions()

    def _add_single_actions(self) -> None:
        for c in self._all_54_cards():
            self._add_action(Meld("single", (c,), 1, rank_of(c), suit_of(c)))

    def _add_group_actions(self) -> None:
        """
        グループアクション（同ランク）を生成
        数札 + ジョーカーの組み合わせをすべて生成
        """
        for rank in RANKS:
            naturals = [f"{s}{rank}" for s in SUITS]
            for size in [2, 3, 4]:
                for joker_count in range(0, min(2, size) + 1):
                    natural_count = size - joker_count
                    if natural_count < 0 or natural_count > 4:
                        continue
                    
                    if natural_count == 0:
                        # ジョーカーのみのグループ
                        if joker_count <= len(JOKERS):
                            from itertools import combinations as comb_iter
                            for joker_comb in comb_iter(JOKERS, joker_count):
                                cards = tuple(sorted(list(joker_comb)))
                                # ジョーカーのみの場合は top_rank="JOKER"
                                self._add_action(Meld("group", cards, size, "JOKER", None))
                    else:
                        # 数札 + ジョーカーの組み合わせ
                        for natural_comb in combinations(naturals, natural_count):
                            # ジョーカーのすべての組み合わせで生成
                            if joker_count == 0:
                                joker_combs = [()]
                            else:
                                from itertools import combinations as comb_iter
                                joker_combs = list(comb_iter(JOKERS, joker_count))
                            
                            for joker_comb in joker_combs:
                                cards = tuple(sorted(list(natural_comb) + list(joker_comb)))
                                self._add_action(Meld("group", cards, size, rank, None))

    def _add_straight_actions(self) -> None:
        for suit in SUITS:
            suited_cards = [f"{suit}{r}" for r in RANKS]
            for length in range(3, 14):
                for start in range(0, 14 - length):
                    window = suited_cards[start : start + length]
                    top_rank = RANKS[start + length - 1]
                    self._add_action(Meld("straight", tuple(window), length, top_rank, suit))

                    # ジョーカー1枚置換
                    for miss in range(length):
                        cards = [c for i, c in enumerate(window) if i != miss] + [JOKERS[0]]
                        self._add_action(Meld("straight", tuple(sorted(cards)), length, top_rank, suit))

                    # ジョーカー2枚置換
                    for miss1, miss2 in combinations(range(length), 2):
                        cards = [c for i, c in enumerate(window) if i not in (miss1, miss2)] + JOKERS
                        self._add_action(Meld("straight", tuple(sorted(cards)), length, top_rank, suit))

    def _add_action(self, action: Meld | str) -> None:
        key = self._canon(action)
        if key in self.action_index:
            return
        self.action_index[key] = len(self.action_table)
        self.action_table.append(action)

    @staticmethod
    def _canon(action: Meld | str) -> tuple:
        if action == PASS:
            return (PASS,)
        return (action.meld_type, tuple(sorted(action.cards)))

    @staticmethod
    def _all_54_cards() -> list[str]:
        return [f"{s}{r}" for s in SUITS for r in RANKS] + JOKERS

