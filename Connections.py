# -*- coding: utf-8 -*-
# ==============================================================
# シナプス結合モデル（全結合および遅延結合）
# --------------------------------------------------------------
# このスクリプトでは、ニューラルネットワークの接続構造を表現する
# 2種類のクラスを定義する：
#   1. FullConnection : 標準的な全結合（行列積による線形結合）
#   2. DelayConnection : シナプス伝達遅延を考慮した接続（履歴バッファ）
#
# これらはスパイキングニューラルネットワーク（SNN）や
# 時系列処理モデルにおいて基本となる構成要素である。
# ==============================================================

import numpy as np


# ==============================================================
# 【1】全結合クラス（FullConnection）
# --------------------------------------------------------------
# 入力ベクトル x に対して出力 y = W x を計算する。
# 必要に応じて、初期重み initW を外部から指定可能。
# ==============================================================
class FullConnection:
    def __init__(self, N_in, N_out, initW=None):
        """
        Args:
            N_in (int): 入力ニューロン数
            N_out (int): 出力ニューロン数
            initW (ndarray): 初期重み行列 (shape: [N_out, N_in])
        """
        if initW is not None:
            # 外部で指定された初期値を使用
            self.W = initW
        else:
            # 0〜0.1の一様乱数で初期化
            self.W = 0.1 * np.random.rand(N_out, N_in)

    def backward(self, x):
        """
        逆方向伝播（出力側から入力側への信号伝達）

        Args:
            x (ndarray): 出力側の信号 (shape: [N_out])

        Returns:
            ndarray: 入力側への逆伝達信号 (shape: [N_in])
        """
        return np.dot(self.W.T, x)  # self.W.T @ x と同等

    def __call__(self, x):
        """
        順方向伝播（入力から出力への信号伝達）

        Args:
            x (ndarray): 入力信号 (shape: [N_in])

        Returns:
            ndarray: 出力信号 (shape: [N_out])
        """
        return np.dot(self.W, x)  # self.W @ x と同等


# ==============================================================
# 【2】遅延結合クラス（DelayConnection）
# --------------------------------------------------------------
# 入力信号 x を一定時間遅らせて出力する。
# 内部に履歴バッファ（state）を持ち、時間遅延をシミュレート。
# これは生理学的シナプス伝達の「伝達遅延」を模倣する。
# ==============================================================
class DelayConnection:
    def __init__(self, N, delay, dt=1e-4):
        """
        Args:
            N (int): ニューロン数（バッファサイズ）
            delay (float): 遅延時間 [s]
            dt (float): シミュレーション時間刻み [s]
        """
        self.N = N
        self.nt_delay = round(delay / dt)  # 遅延をステップ数に換算
        self.state = np.zeros((N, self.nt_delay))  # 過去の入力を保存するバッファ

    def initialize_states(self):
        """内部状態（遅延バッファ）をゼロで初期化"""
        self.state = np.zeros((self.N, self.nt_delay))

    def __call__(self, x):
        """
        遅延付き出力を生成

        Args:
            x (ndarray): 現在の入力信号 (shape: [N])

        Returns:
            ndarray: delay 秒前の入力信号 (shape: [N])
        """
        out = self.state[:, -1]  # 一番古い入力を出力として返す（遅延時間分の出力）

        # バッファを1ステップ分右にシフト（古い入力を後ろへ）
        self.state[:, 1:] = self.state[:, :-1]

        # 現在の入力をバッファの先頭（時刻0）に格納
        self.state[:, 0] = x

        return out
