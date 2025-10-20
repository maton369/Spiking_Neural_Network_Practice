# -*- coding: utf-8 -*-
# ==============================================================
# Leaky RNN (PyTorch実装)
# --------------------------------------------------------------
# このコードは、ChainerベースのleakyRNNユニットを
# PyTorchで等価に書き換えたものです。
# 各時刻で「リーキー統合」とReLU非線形性を組み合わせ、
# RNNの時間的平滑性（緩やかな変化）を再現します。
# ノイズ項 (sigma_rec) により、生理的ゆらぎを導入可能です。
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyRNN(nn.Module):
    def __init__(self, inp=32, mid=128, alpha=0.2, sigma_rec=0.1, device="cuda"):
        """
        Leaky RNN Unit (PyTorch version)

        Args:
            inp (int): 入力ユニット数
            mid (int): 隠れ層ユニット数
            alpha (float): リーク率 (dt / τ)
            sigma_rec (float): リカレントノイズの標準偏差
            device (str): 使用デバイス（'cuda'または'cpu'）

        Example:
            >>> rnn = LeakyRNN()
            >>> rnn.reset_state()
            >>> x = torch.ones((1, 32)).float().to('cuda')
            >>> y = rnn(x)
            >>> y.shape
            torch.Size([1, 128])
        """
        super(LeakyRNN, self).__init__()

        # 前向き入力重みとリカレント重みを定義
        self.Wx = nn.Linear(inp, mid)
        self.Wr = nn.Linear(mid, mid, bias=False)

        # モデルパラメータ
        self.inp = inp
        self.mid = mid
        self.alpha = alpha
        self.sigma_rec = sigma_rec
        self.device = device

        # 隠れ状態 r（リーキー統合された活動）を初期化
        self.r = None

    # ----------------------------------------------------------
    # 状態リセット関数
    # ----------------------------------------------------------
    def reset_state(self, r=None):
        """
        隠れ状態をリセットする。
        r が指定されない場合は None として初期化される。
        """
        self.r = r

    # ----------------------------------------------------------
    # 状態初期化関数
    # ----------------------------------------------------------
    def initialize_state(self, batch_size):
        """
        隠れ状態rをゼロベクトルで初期化。
        Args:
            batch_size (int): 入力のバッチサイズ
        """
        self.r = torch.zeros(batch_size, self.mid, dtype=torch.float32).to(self.device)

    # ----------------------------------------------------------
    # 順伝播処理
    # ----------------------------------------------------------
    def forward(self, x):
        """
        RNNの1ステップ順伝播。
        入力xと前回状態rを統合し、リーキー項 + ノイズ項を加味して更新する。

        Args:
            x (Tensor): shape = (batch_size, inp)

        Returns:
            Tensor: 更新後の隠れ状態 r (shape = [batch_size, mid])
        """
        # 状態が未初期化の場合は初期化
        if self.r is None:
            self.initialize_state(x.shape[0])

        # リカレント入力 Wr*r と フィードフォワード入力 Wx*x
        z = self.Wr(self.r) + self.Wx(x)

        # リカレントノイズを加える（sigma_rec > 0 の場合）
        if self.sigma_rec is not None and self.sigma_rec > 0:
            noise = torch.randn_like(z) * self.sigma_rec
            z = z + noise

        # リーク付き更新式：
        # r_new = (1 - α) * r_old + α * ReLU(z)
        r_new = (1 - self.alpha) * self.r + self.alpha * F.relu(z)

        # 状態を更新
        self.r = r_new

        return r_new


# ==============================================================
# ✅ 動作テスト
# --------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = LeakyRNN(inp=32, mid=128, alpha=0.2, sigma_rec=0.1, device=device).to(device)

    # 入力テンソル生成
    x = torch.ones((1, 32)).float().to(device)

    # 順伝播を実行
    y = net(x)
    print("出力テンソル形状:", y.shape)
