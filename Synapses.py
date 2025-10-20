# -*- coding: utf-8 -*-
# ==============================================================
# シナプス電流モデル（クラス実装）
# --------------------------------------------------------------
# このスクリプトでは、生物学的シナプス電流の2種類の時間特性モデルを定義する：
# 1. SingleExponentialSynapse : 単一指数関数型シナプス
# 2. DoubleExponentialSynapse : 二重指数関数型シナプス
#
# これらは、神経ネットワークにおけるシナプス伝達（ポストシナプス電流, PSC）の
# 時間発展を数値的にシミュレーションするための基礎モデルである。
# ==============================================================

import numpy as np

# ==============================================================
# 【1】単一指数関数型シナプスモデル
# --------------------------------------------------------------
# モデル式：
#   dr/dt = -r/td + spike/td
#
# - spike: シナプス前ニューロンの発火入力（1 または 0）
# - td   : シナプス電流の減衰時定数
#
# 特徴：
#   簡単な指数減衰による応答。
#   スパイク直後に最大値を取り、単調に減衰していく。
# ==============================================================


class SingleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=5e-3):
        """
        Args:
            N (int): ニューロン数
            dt (float): シミュレーション時間刻み [s]
            td (float): シナプス電流の減衰時定数 [s]
        """
        self.N = N  # ニューロン数
        self.dt = dt  # 時間刻み
        self.td = td  # 減衰時定数
        self.r = np.zeros(N)  # シナプス電流の状態ベクトル

    def initialize_states(self):
        """シナプス状態をゼロ初期化"""
        self.r = np.zeros(self.N)

    def __call__(self, spike):
        """
        スパイク入力を受けてシナプス電流を更新する。

        Args:
            spike (ndarray): shape = (N,), スパイク入力（0または1）

        Returns:
            ndarray: 更新後のシナプス電流 r
        """
        # オイラー法による時間発展
        # r(t+dt) = r(t)*(1 - dt/td) + spike/td
        r = self.r * (1 - self.dt / self.td) + spike / self.td

        # 状態を更新
        self.r = r
        return r


# ==============================================================
# 【2】二重指数関数型シナプスモデル
# --------------------------------------------------------------
# モデル式：
#   dr/dt  = -r/tr + h_r
#   dh_r/dt = -h_r/td + spike / (tr * td)
#
# - tr : 立ち上がり時定数
# - td : 減衰時定数
#
# 特徴：
#   シナプス電流が急上昇してからゆっくり減衰する二相性の応答を再現。
#   実際の神経シナプス伝達のダイナミクスに近い。
# ==============================================================


class DoubleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            N (int): ニューロン数
            dt (float): シミュレーション時間刻み [s]
            td (float): シナプス電流の減衰時定数 [s]
            tr (float): シナプス電流の立ち上がり時定数 [s]
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros(N)  # シナプス電流
        self.hr = np.zeros(N)  # 上昇成分（中間変数）

    def initialize_states(self):
        """シナプス状態をゼロ初期化"""
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)

    def __call__(self, spike):
        """
        スパイク入力を受けて二重指数型シナプス電流を更新する。

        Args:
            spike (ndarray): shape = (N,), スパイク入力（0または1）

        Returns:
            ndarray: 更新後のシナプス電流 r
        """
        # オイラー法による数値積分
        # r(t+dt) = r(t)*(1 - dt/tr) + h_r(t)*dt
        # h_r(t+dt) = h_r(t)*(1 - dt/td) + spike/(tr*td)
        r = self.r * (1 - self.dt / self.tr) + self.hr * self.dt
        hr = self.hr * (1 - self.dt / self.td) + spike / (self.tr * self.td)

        # 状態更新
        self.r = r
        self.hr = hr

        return r
