"""
AlphaZero MCTS (Monte Carlo Tree Search) Implementation
PUCT algorithm for Daifugo
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch


@dataclass
class MCTSNode:
    """MCTSノード"""
    state_key: str  # 状態のハッシュキー
    parent: Optional['MCTSNode'] = None
    children: Dict[int, 'MCTSNode'] = None  # action_id -> child
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy Networkからの事前確率
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    @property
    def value(self) -> float:
        """平均価値"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return not self.is_expanded

    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'MCTSNode']:
        """PUCTアルゴリズムで子ノードを選択"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action_id, child in self.children.items():
            # PUCTスコア計算
            q_value = child.value if child.visit_count > 0 else 0.0
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action_id
                best_child = child

        return best_action, best_child

    def expand(self, legal_actions: List[int], policy_probs: np.ndarray):
        """ノードを展開"""
        self.is_expanded = True

        for action_id in legal_actions:
            if action_id not in self.children:
                # Policy確率を事前確率として設定
                prior = policy_probs[action_id] if action_id < len(policy_probs) else 0.0
                child = MCTSNode(
                    state_key="",  # 後で設定
                    parent=self,
                    prior=prior
                )
                self.children[action_id] = child

    def backup(self, value: float):
        """バックアップ: 葉ノードから根まで価値を伝播"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value  # 交代で符号反転


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, c_puct: float = 1.0, max_simulations: int = 800):
        self.c_puct = c_puct
        self.max_simulations = max_simulations
        self.root: Optional[MCTSNode] = None
        self.node_cache: Dict[str, MCTSNode] = {}  # 状態キー -> ノード

    def get_state_key(self, state) -> str:
        """状態のハッシュキー生成"""
        # 簡易的なハッシュ（実際にはもっと詳細に）
        hands_str = f"{sorted(state.hands[0])}_{sorted(state.hands[1])}"
        table_str = str(state.table) if state.table else "None"
        turn_str = str(state.turn)
        return f"{hands_str}_{table_str}_{turn_str}_{state.revolution}"

    def search(self, env, policy_net, value_net) -> np.ndarray:
        """
        MCTS探索を実行し、アクション確率を返す

        Args:
            env: ゲーム環境
            policy_net: Policy Network
            value_net: Value Network

        Returns:
            action_probs: アクション確率分布
        """
        # ルートノードの初期化
        state_key = self.get_state_key(env.state)
        if state_key not in self.node_cache:
            self.node_cache[state_key] = MCTSNode(state_key)
        self.root = self.node_cache[state_key]

        # シミュレーション実行
        for _ in range(self.max_simulations):
            self._simulate(env, policy_net, value_net)

        # アクション確率を計算（訪問回数ベース）
        action_probs = np.zeros(env.action_space.n)
        total_visits = sum(child.visit_count for child in self.root.children.values())

        if total_visits > 0:
            for action_id, child in self.root.children.items():
                action_probs[action_id] = child.visit_count / total_visits

        return action_probs

    def _simulate(self, env, policy_net, value_net):
        """1回のシミュレーション実行"""
        # コピー環境を作成（元の環境を変更しない）
        sim_env = self._copy_env(env)
        path = []  # 探索パス
        node = self.root

        # Selection: 葉ノードまで降下
        while not node.is_leaf():
            action_id, child = node.select_child(self.c_puct)
            if child is None:
                break

            # 環境を進める
            obs, reward, done, _, info = sim_env.step(action_id)
            path.append((node, action_id))

            # 子ノードの状態キーを設定
            if not child.state_key:
                child.state_key = self.get_state_key(sim_env.state)
                if child.state_key not in self.node_cache:
                    self.node_cache[child.state_key] = child

            node = child

            if done:
                break

        # Expansion & Evaluation
        if not sim_env.state.done:
            # 現在の観測を取得
            current_obs = sim_env.env.encode_obs(sim_env.state, sim_env.player)
            
            # 合法アクションを取得
            legal_actions = sim_env.env.legal_actions(sim_env.state)
            legal_ids = [sim_env.env.action_index[sim_env.env._canon(a)] for a in legal_actions]

            # Policy Networkでアクション確率を予測
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
            policy_logits = policy_net(obs_tensor)
            policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).detach().numpy()

            # ノードを展開
            node.expand(legal_ids, policy_probs)

            # Value Networkで葉ノードの価値を評価
            value_out = value_net(obs_tensor)
            leaf_value = value_out.item()
        else:
            # ゲーム終了時は報酬から価値を決定
            reward = sim_env.state.last_winner if sim_env.state.last_winner == 0 else -1
            leaf_value = reward

        # Backup: 価値を伝播
        for node, action_id in reversed(path):
            node.backup(leaf_value)

    def _copy_env(self, env):
        """環境のコピーを作成"""
        # 簡易的なコピー（実際の実装ではもっと深くコピーする必要がある）
        from wrapper import DaifugoGymEnv
        copy_env = DaifugoGymEnv(seed=42)  # 同じシードでコピー
        copy_env.state = env.state  # 状態をコピー
        copy_env.player = env.player
        copy_env.obs, copy_env.info = copy_env.reset()  # resetを呼ぶ
        return copy_env

    def get_best_action(self) -> int:
        """最も訪問されたアクションを返す"""
        if not self.root or not self.root.children:
            return 0

        best_action = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_action