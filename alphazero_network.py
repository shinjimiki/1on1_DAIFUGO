"""
AlphaZero Neural Network for Daifugo
Policy Network + Value Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """Policy Network: 状態からアクション確率を予測"""

    def __init__(self, input_size: int = 146, hidden_size: int = 256, action_size: int = 7052):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Policy確率を出力"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.policy_head(x)
        return logits


class ValueNetwork(nn.Module):
    """Value Network: 状態から勝敗確率を予測"""

    def __init__(self, input_size: int = 146, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Value値(-1~1)を出力"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = torch.tanh(self.value_head(x))
        return value


class AlphaZeroNet(nn.Module):
    """AlphaZero Network: Policy + Value"""

    def __init__(self, input_size: int = 146, hidden_size: int = 256, action_size: int = 7052):
        super().__init__()
        self.policy_net = PolicyNetwork(input_size, hidden_size, action_size)
        self.value_net = ValueNetwork(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Policy確率とValue値を出力"""
        policy_logits = self.policy_net(x)
        value = self.value_net(x)
        return policy_logits, value

    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推論モードでの予測"""
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(obs)
            policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value