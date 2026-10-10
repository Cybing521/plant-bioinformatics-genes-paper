"""1D-CNN 基线：DeepSEA 风格（堆叠卷积+池化）与 Basenji 风格（残差块）。"""
import math

import torch
import torch.nn as nn


class DeepSEABlock(nn.Module):
    def __init__(self, cin, cout, kernel_size, pool_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, padding=pad), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, kernel_size, padding=pad), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.MaxPool1d(pool_size, padding=pool_size // 2),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Basenji 风格残差卷积块。"""

    def __init__(self, cin, cout, kernel_size, dil=1):
        super().__init__()
        pad = ((kernel_size - 1) * dil) // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel_size, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = None if cin == cout else nn.Conv1d(cin, cout, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.skip is not None:
            x = self.skip(x)
        return self.act(h + x)


class SequenceCNN(nn.Module):
    """输入 (N,4,L)，输出 logits (N, num_classes)。

    style: "deepsea"（卷积+池化+FC）或 "basenji"（残差块+全局池化+FC）
    """

    def __init__(self, input_len, n_channels=4, num_classes=1, task="binary",
                 style="deepsea", conv_channels=(256, 256, 256),
                 kernel_size=8, pool_size=4, fc_dim=256):
        super().__init__()
        self.task = task
        if style == "deepsea":
            blocks, cin = [], n_channels
            for cout in conv_channels:
                blocks.append(DeepSEABlock(cin, cout, kernel_size, pool_size))
                cin = cout
            self.encoder = nn.Sequential(*blocks)
        else:  # basenji
            dil = 1
            blocks, cin = [], n_channels
            for cout in conv_channels:
                blocks.append(ResConvBlock(cin, cout, kernel_size, dil))
                blocks.append(nn.MaxPool1d(2))
                cin, dil = cout, dil * 2
            self.encoder = nn.Sequential(*blocks)

        # 自适应池化 -> 任意输入长度均可用（PGB 6000bp / 自建 ~3000bp）
        # 用全局最大池化：检测"模体是否出现"，比平均池化保留信号
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_channels[-1], fc_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


def build_cnn(cfg: dict, num_classes=1, task="binary") -> SequenceCNN:
    m = cfg["model"]["cnn"]
    return SequenceCNN(
        input_len=m["input_len"],
        num_classes=num_classes,
        task=task,
        style=m.get("style", "deepsea"),
        conv_channels=tuple(m["conv_channels"]),
        kernel_size=m["kernel_size"],
        pool_size=m["pool_size"],
        fc_dim=m["fc_dim"],
    )
