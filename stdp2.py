# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:52:03 2019
@author: user
"""
# ==============================================================
# オンラインSTDP（Spike-Timing Dependent Plasticity）の逐次更新モデル
# --------------------------------------------------------------
# 本コードでは、スパイクの発生タイミング差に基づくシナプス可塑性を
# 「逐次更新（online）」で計算する。
# STDPは、スパイク順序によってシナプス強度（重み）が変化する学習則であり、
# pre → post の順でスパイクが発火した場合はシナプスが強化され（LTP）、
# post → pre の順で発火した場合はシナプスが弱化される（LTD）。
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

# 乱数シード（再現性の確保）
np.random.seed(seed=0)

# ==============================================================
# シミュレーション定数の設定
# --------------------------------------------------------------
# dt : 時間刻み幅 [s]
# T  : シミュレーション時間 [s]
# nt : ステップ数
# tau_p, tau_m : LTP/LTD の時定数 [s]
# A_p, A_m : シナプス可塑性の学習係数
# ==============================================================
dt = 1e-3  # sec（=1 ms）
T = 0.5  # sec（=500 ms）
nt = round(T / dt)

tau_p = tau_m = 2e-2  # 20 ms
A_p = 0.01  # LTPの係数
A_m = 1.05 * A_p  # LTDの係数（LTPよりわずかに大きい）

# ==============================================================
# スパイク系列の設定
# --------------------------------------------------------------
# spike_pre : シナプス前ニューロンのスパイク系列
# spike_post: シナプス後ニューロンのスパイク系列
# （固定的なスパイクタイミングを用いて学習挙動を観察）
# ==============================================================
spike_pre = np.zeros(nt)
spike_pre[[50, 200, 225, 300, 425]] = 1  # preニューロンの発火時刻 [ms単位]

spike_post = np.zeros(nt)
spike_post[[100, 150, 250, 350, 400]] = 1  # postニューロンの発火時刻

# ==============================================================
# 記録用配列の初期化
# --------------------------------------------------------------
# x_pre_arr, x_post_arr : pre/postニューロンのスパイクトレースの履歴
# w_arr : シナプス重みの変化履歴
# ==============================================================
x_pre_arr = np.zeros(nt)
x_post_arr = np.zeros(nt)
w_arr = np.zeros(nt)

# ==============================================================
# 状態変数の初期化
# --------------------------------------------------------------
# x_pre, x_post : pre/postニューロンのトレース変数
# w : シナプス重み
# ==============================================================
x_pre = 0.0
x_post = 0.0
w = 0.0

# ==============================================================
# オンラインSTDP学習ループ
# --------------------------------------------------------------
# 各時刻 t で以下を実行：
#   1. pre/postスパイクトレースの更新（指数減衰 + 発火入力）
#   2. LTP/LTDに基づく重み変化 dw の計算
#   3. 重み w の更新と記録
# ==============================================================
for t in range(nt):
    # pre-synaptic trace の更新（指数減衰＋スパイクによる上昇）
    x_pre = x_pre * (1 - dt / tau_p) + spike_pre[t]

    # post-synaptic trace の更新
    x_post = x_post * (1 - dt / tau_m) + spike_post[t]

    # 重み変化の計算
    # LTP成分：A_p * x_pre * spike_post
    # LTD成分：A_m * x_post * spike_pre
    dw = A_p * x_pre * spike_post[t] - A_m * x_post * spike_pre[t]

    # シナプス重みの更新
    w += dw

    # 各値を記録
    x_pre_arr[t] = x_pre
    x_post_arr[t] = x_post
    w_arr[t] = w

# ==============================================================
# 可視化
# --------------------------------------------------------------
# 上から順に：
#   1. preニューロンのトレース
#   2. preニューロンのスパイク列
#   3. postニューロンのスパイク列
#   4. postニューロンのトレース
#   5. シナプス重みの推移
# ==============================================================
time = np.arange(nt) * dt * 1e3  # 時間軸（ms単位）


def hide_ticks():
    """上と右の軸を非表示にする補助関数"""
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")


plt.figure(figsize=(6, 6))

# preニューロンのトレース
plt.subplot(5, 1, 1)
plt.plot(time, x_pre_arr, color="k")
plt.ylabel("$x_{pre}$")
hide_ticks()
plt.xticks([])

# preニューロンのスパイク系列
plt.subplot(5, 1, 2)
plt.plot(time, spike_pre, color="k")
plt.ylabel("pre-spikes")
hide_ticks()
plt.xticks([])

# postニューロンのスパイク系列
plt.subplot(5, 1, 3)
plt.plot(time, spike_post, color="k")
plt.ylabel("post-spikes")
hide_ticks()
plt.xticks([])

# postニューロンのトレース
plt.subplot(5, 1, 4)
plt.plot(time, x_post_arr, color="k")
plt.ylabel("$x_{post}$")
hide_ticks()
plt.xticks([])

# シナプス重みの変化
plt.subplot(5, 1, 5)
plt.plot(time, w_arr, color="k")
plt.xlabel("$t$ (ms)")
plt.ylabel("Synaptic weight $w$")
hide_ticks()
plt.tight_layout()
plt.savefig("online_stdp.pdf")

# ==============================================================
# 💬 コメント
# --------------------------------------------------------------
# - preニューロンのスパイク後、x_preが急上昇し指数的に減衰する。
# - postニューロンのスパイク時に、pre側のトレースが大きければLTP（重み上昇）、
#   逆にpost側トレースが大きければLTD（重み減少）が生じる。
# - この結果、スパイクの順序に基づく因果関係がシナプス強度として学習される。
# ==============================================================
