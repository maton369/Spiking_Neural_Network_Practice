# -*- coding: utf-8 -*-
"""
FORCE学習 (First-Order Reduced and Controlled Error) による
リカレントスパイキングニューラルネットワーク (RSNN) の学習・可視化スクリプト

-------------------------------------------------------------
本スクリプトでは Izhikevich モデルニューロンを多数 (N=2000) 用いた
リカレントスパイキングネットワークを構築し、
FORCE学習 (Sussillo & Abbott, 2009) により
教師信号（正弦波）を出力するように訓練する。

主な流れ:
1. 教師信号 (sin波) の生成
2. ニューロン群とシナプスの初期化
3. 再帰結合行列 OMEGA のスパース初期化
4. FORCE (RLS) による出力デコーダ Φ の更新
5. 結果として学習前後の発火活動と出力を可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# --- モデル定義に必要なクラスをインポート ---
from Neurons import IzhikevichNeuron
from Synapses import DoubleExponentialSynapse

np.random.seed(seed=0)

#################
## モデルの定義 ##
#################
N = 2000  # ニューロン数（リカレント層）
dt = 0.04  # タイムステップ [ms]

# --- Izhikevichニューロンパラメータ ---
C = 250  # 膜容量 (pF)
a = 0.01  # 回復変数uの時間スケール (1/ms)
b = -2  # vに対する共鳴係数
k = 2.5  # 電位-電流間ゲイン
d = 200  # 発火後のuへのリセット項 (pA)
vrest = -60  # 静止膜電位 (mV)
vreset = -65  # リセット電位 (mV)
vthr = -20  # 閾値電位 (mV)
vpeak = 30  # ピーク電位 (mV)

# --- シナプス特性 ---
td = 20  # 衰退時定数 [ms]
tr = 2  # 立ち上がり時定数 [ms]

# --- FORCE学習の初期化 ---
P = np.eye(N) * 2  # 相関行列の逆行列 (RLS初期化)

# --- 教師信号 (sin波) ---
T = 15000  # シミュレーション時間 [ms]
tmin = round(5000 / dt)  # 重み更新開始ステップ
tcrit = round(10000 / dt)  # 重み更新終了ステップ
step = 50  # 重み更新間隔
nt = round(T / dt)  # シミュレーション総ステップ
Q = 5e3
G = 5e3  # スケーリング定数
zx = np.sin(2 * math.pi * np.arange(nt) * dt * 5 * 1e-3)  # 教師信号（5Hzのsin波）

# --- ニューロンとシナプスの定義 ---
neurons = IzhikevichNeuron(
    N=N,
    dt=dt,
    C=C,
    a=a,
    b=b,
    k=k,
    d=d,
    vrest=vrest,
    vreset=vreset,
    vthr=vthr,
    vpeak=vpeak,
)
neurons.v = vrest + np.random.rand(N) * (vpeak - vrest)  # 初期膜電位をランダムに設定

# 出力・再帰シナプスの生成
synapses_out = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)
synapses_rec = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)

# --- 再帰重み行列の初期化 ---
p = 0.1  # ネットワークのスパース率
OMEGA = G * (np.random.randn(N, N)) * (np.random.rand(N, N) < p) / (math.sqrt(N) * p)

# 行ごとの総和が0になるように正規化（発散防止）
for i in range(N):
    QS = np.where(np.abs(OMEGA[i, :]) > 0)[0]
    OMEGA[i, QS] -= np.sum(OMEGA[i, QS], axis=0) / len(QS)

# --- FORCE学習用の初期化 ---
k = 1  # 出力次元数
E = (2 * np.random.rand(N, k) - 1) * Q  # 出力投影行列
PSC = np.zeros(N)  # シナプス後電流
JD = np.zeros(N)  # 再帰入力の重み和
z = np.zeros(k)  # 出力
Phi = np.zeros(N)  # 学習される出力重み

# --- 記録用変数 ---
REC_v = np.zeros((nt, 10))  # 膜電位記録（10ニューロン分）
current = np.zeros(nt)  # 出力電流記録
tspike = np.zeros((5 * nt, 2))  # スパイク時刻 (index, 時間)
ns = 0  # 総スパイク数

BIAS = 1000  # 定常入力バイアス電流

#################
## シミュレーション ##
#################
for t in tqdm(range(nt)):
    # --- 入力電流の計算 ---
    I = PSC + np.dot(E, z) + BIAS  # シナプス後電流 + 出力入力 + バイアス

    # --- ニューロンの発火 ---
    s = neurons(I)  # スパイクを計算 (Izhikevichモデル)

    # --- 発火したニューロンのインデックスを取得 ---
    index = np.where(s)[0]
    len_idx = len(index)
    if len_idx > 0:
        # 発火ニューロンからの再帰入力を合計
        JD = np.sum(OMEGA[:, index], axis=1)
        tspike[ns : ns + len_idx, :] = np.vstack((index, np.full(len_idx, dt * t))).T
        ns += len_idx  # スパイク数をカウント

    # --- シナプス後電流の更新 ---
    PSC = synapses_rec(JD * (len_idx > 0))

    # --- 出力電流（神経伝達物質放出量）---
    r = synapses_out(s)
    r = np.expand_dims(r, 1)  # (N,) → (N,1)

    # --- デコードされた出力 ---
    z = Phi.T @ r
    err = z - zx[t]  # 教師信号との差分

    # --- FORCE学習（RLSによる出力重み更新） ---
    if t % step == 1 and tmin < t < tcrit:
        cd = P @ r
        Phi -= cd @ err.T
        P -= (cd @ cd.T) / (1.0 + r.T @ cd)

    # --- 記録 ---
    current[t] = z
    REC_v[t] = neurons.v_[:10]

#################
#### 結果表示 ####
#################
TotNumSpikes = ns
M = tspike[tspike[:, 1] > dt * tcrit, :]
AverageRate = len(M) / (N * (T - dt * tcrit)) * 1e3
print("\nTotal number of spikes :", TotNumSpikes)
print("Average firing rate(Hz):", AverageRate)


# --- 可視化用関数 ---
def hide_ticks():
    """上・右軸を非表示にし、見やすいグラフにする"""
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")


# --- 発火パターンの可視化（学習前後比較） ---
step_range = 20000
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for j in range(5):
    plt.plot(
        np.arange(step_range) * dt * 1e-3,
        REC_v[:step_range, j] / (50 - vreset) + j,
        color="k",
    )
hide_ticks()
plt.title("Pre-Learning")
plt.xlabel("Time (s)")
plt.ylabel("Neuron Index")

plt.subplot(1, 2, 2)
for j in range(5):
    plt.plot(
        np.arange(nt - step_range, nt) * dt * 1e-3,
        REC_v[nt - step_range :, j] / (50 - vreset) + j,
        color="k",
    )
hide_ticks()
plt.title("Post Learning")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("Iz_FORCE_prepost.pdf")
plt.show()

# --- 出力波形の可視化（教師信号との比較） ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(nt) * dt * 1e-3, zx, label="Target", color="k")
plt.plot(
    np.arange(nt) * dt * 1e-3,
    current,
    label="Decoded output",
    linestyle="dashed",
    color="k",
)
plt.xlim(4.5, 5.5)
plt.ylim(-1.1, 1.4)
hide_ticks()
plt.title("Pre/peri Learning")
plt.xlabel("Time (s)")
plt.ylabel("current")

plt.subplot(1, 2, 2)
plt.plot(np.arange(nt) * dt * 1e-3, zx, label="Target", color="k")
plt.plot(
    np.arange(nt) * dt * 1e-3,
    current,
    label="Decoded output",
    linestyle="dashed",
    color="k",
)
plt.xlim(14, 15)
plt.ylim(-1.1, 1.4)
hide_ticks()
plt.title("Post Learning")
plt.xlabel("Time (s)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("Iz_FORCE_decoded.pdf")
plt.show()


# --- （オプション）重み行列の固有値可視化 ---
Z = np.linalg.eig(OMEGA + np.expand_dims(E,1) @ np.expand_dims(Phi,1).T)
Z2 = np.linalg.eig(OMEGA)
plt.figure(figsize=(6, 5))
plt.title('Weight eigenvalues')
plt.scatter(Z2[0].real, Z2[0].imag, c='r', s=5, label='Pre-Learning')
plt.scatter(Z[0].real, Z[0].imag, c='k', s=5, label='Post-Learning')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
