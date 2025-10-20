# -*- coding: utf-8 -*-
# ==============================================================
# SuperSpike学習則によるスパイキングネットワーク学習デモ
# --------------------------------------------------------------
# このコードは、SuperSpike論文（Zenke & Ganguli, 2018）に基づき、
# 誤差信号とエリジビリティトレースを用いたSNN（Spiking Neural Network）の
# 局所学習則をPythonで実装したもの。
#
# ネットワーク構成：
#   入力層（50ユニット, Poissonスパイク）
#   中間層（4ユニット, Current-based LIF）
#   出力層（1ユニット, Current-based LIF）
#
# 学習では、出力スパイクが教師信号（target spike）に近づくように
# シナプス重みを更新していく。
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Neurons import CurrentBasedLIF
from Synapses import DoubleExponentialSynapse
from Connections import FullConnection, DelayConnection

np.random.seed(seed=0)


# ==============================================================
# 誤差信号 (Error Signal)
# --------------------------------------------------------------
# 出力スパイクと教師信号の差分から誤差を生成。
# 二重指数関数フィルタにより時間方向に滑らかに伝搬させる。
# ==============================================================
class ErrorSignal:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt  # 時間刻み
        self.td = td  # 減衰時定数
        self.tr = tr  # 立ち上がり時定数
        self.N = N  # 出力ユニット数
        self.r = np.zeros(N)  # 誤差信号のフィルタ出力
        self.hr = np.zeros(N)  # 補助変数（2次系用）
        self.b = (td / tr) ** (td / (tr - td))  # 正規化定数（安定化用）

    def initialize_states(self):
        """誤差信号状態の初期化"""
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)

    def __call__(self, output_spike, target_spike):
        """誤差信号の更新"""
        r = self.r * (1 - self.dt / self.tr) + self.hr / self.td * self.dt
        hr = self.hr * (1 - self.dt / self.td) + (target_spike - output_spike) / self.b

        self.r = r
        self.hr = hr

        return r


# ==============================================================
# エリジビリティトレース (Eligibility Trace)
# --------------------------------------------------------------
# シナプス前後の活動相関（pre × post）を二重指数関数で追跡。
# これが局所的な「勾配情報」を担い、誤差項と掛け合わせて重みを更新。
# ==============================================================
class EligibilityTrace:
    def __init__(self, N_in, N_out, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt
        self.td = td
        self.tr = tr
        self.N_in = N_in
        self.N_out = N_out
        self.r = np.zeros((N_out, N_in))  # トレース変数
        self.hr = np.zeros((N_out, N_in))  # 補助変数

    def initialize_states(self):
        """トレース変数の初期化"""
        self.r = np.zeros((self.N_out, self.N_in))
        self.hr = np.zeros((self.N_out, self.N_in))

    def surrogate_derivative_fastsigmoid(self, u, beta=1, vthr=-50):
        """スパイク非線形の近似微分（fast sigmoid近似）"""
        return 1 / (1 + np.abs(beta * (u - vthr))) ** 2

    def __call__(self, pre_current, post_voltage):
        """トレース更新"""
        # pre: 入力電流, post: 膜電位
        pre_ = np.expand_dims(pre_current, axis=0)
        post_ = np.expand_dims(
            self.surrogate_derivative_fastsigmoid(post_voltage), axis=1
        )

        r = self.r * (1 - self.dt / self.tr) + self.hr * self.dt
        hr = self.hr * (1 - self.dt / self.td) + (post_ @ pre_) / (self.tr * self.td)

        self.r = r
        self.hr = hr

        return r


# ==============================================================
# モデル構築
# --------------------------------------------------------------
# 入力層：Poissonスパイク
# 中間層・出力層：Current-based LIFニューロン
# 接続：FullConnection（重み付き結合）+ DelayConnection（伝達遅延）
# ==============================================================
dt = 1e-4
T = 0.5
nt = round(T / dt)
num_iter = 200

# 重み更新ステップ
t_weight_update = 0.5
nt_b = round(t_weight_update / dt)

# ネットワーク構造
N_in = 50
N_mid = 4
N_out = 1

# 入力Poissonスパイク生成
fr_in = 10
x = np.where(np.random.rand(nt, N_in) < fr_in * dt, 1, 0)

# 教師スパイク列（出力層の理想的な発火タイミング）
y = np.zeros((nt, N_out))
y[int(nt / 10) :: int(nt / 5), :] = 1  # 定期的に発火

# モデル要素の初期化
neurons_1 = CurrentBasedLIF(N_mid, dt=dt)
neurons_2 = CurrentBasedLIF(N_out, dt=dt)
delay_conn1 = DelayConnection(N_in, delay=8e-4)
delay_conn2 = DelayConnection(N_mid, delay=8e-4)
synapses_1 = DoubleExponentialSynapse(N_in, dt=dt)
synapses_2 = DoubleExponentialSynapse(N_mid, dt=dt)
es = ErrorSignal(N_out)
et1 = EligibilityTrace(N_in, N_mid)
et2 = EligibilityTrace(N_mid, N_out)
connect_1 = FullConnection(N_in, N_mid, initW=0.1 * np.random.rand(N_mid, N_in))
connect_2 = FullConnection(N_mid, N_out, initW=0.1 * np.random.rand(N_out, N_mid))

# 学習パラメータ
r0 = 1e-3  # 学習率スケール
gamma = 0.7  # モーメンタム減衰

# 記録配列
current_arr = np.zeros((N_mid, nt))
voltage_arr = np.zeros((N_out, nt))
error_arr = np.zeros((N_out, nt))
lambda_arr = np.zeros((N_out, N_mid, nt))
cost_arr = np.zeros(num_iter)

# ==============================================================
# 学習ループ
# --------------------------------------------------------------
# 各イテレーションごとにスパイク伝播→誤差伝播→重み更新を実行。
# SuperSpike学習則に基づき、誤差信号と局所勾配の積で更新。
# ==============================================================
for i in tqdm(range(num_iter)):
    if i % 15 == 0:
        r0 /= 2  # 学習率減衰

    # 各状態をリセット
    neurons_1.initialize_states()
    neurons_2.initialize_states()
    synapses_1.initialize_states()
    synapses_2.initialize_states()
    delay_conn1.initialize_states()
    delay_conn2.initialize_states()
    es.initialize_states()
    et1.initialize_states()
    et2.initialize_states()

    # オプティマイザ（RMSProp風更新）変数の初期化
    m1 = np.zeros((N_mid, N_in))
    m2 = np.zeros((N_out, N_mid))
    v1 = np.zeros((N_mid, N_in))
    v2 = np.zeros((N_out, N_mid))
    cost = 0
    count = 0

    # --- 時系列シミュレーション ---
    for t in range(nt):
        # フィードフォワード伝播
        c1 = synapses_1(delay_conn1(x[t]))
        h1 = connect_1(c1)
        s1 = neurons_1(h1)

        c2 = synapses_2(delay_conn2(s1))
        h2 = connect_2(c2)
        s2 = neurons_2(h2)

        # 誤差信号の計算と伝播
        e2 = np.expand_dims(es(s2, y[t]), axis=1) / N_out
        e1 = connect_2.backward(e2) / N_mid

        # コストの蓄積（平均二乗誤差）
        cost += np.sum(e2**2)

        # エリジビリティトレースの更新
        lambda2 = et2(c2, neurons_2.v_)
        lambda1 = et1(c1, neurons_1.v_)

        # 勾配の計算
        g2 = e2 * lambda2
        g1 = e1 * lambda1

        # RMSPropライクな更新（過去勾配を指数移動平均）
        v1 = np.maximum(gamma * v1, g1**2)
        v2 = np.maximum(gamma * v2, g2**2)
        m1 += g1
        m2 += g2

        count += 1
        if count == nt_b:
            # --- 重みの更新 ---
            lr1 = r0 / np.sqrt(v1 + 1e-8)
            lr2 = r0 / np.sqrt(v2 + 1e-8)
            dW1 = np.clip(lr1 * m1 * dt, -1e-3, 1e-3)
            dW2 = np.clip(lr2 * m2 * dt, -1e-3, 1e-3)
            connect_1.W = np.clip(connect_1.W + dW1, -0.1, 0.1)
            connect_2.W = np.clip(connect_2.W + dW2, -0.1, 0.1)

            # リセット
            m1 = np.zeros((N_mid, N_in))
            m2 = np.zeros((N_out, N_mid))
            v1 = np.zeros((N_mid, N_in))
            v2 = np.zeros((N_out, N_mid))
            count = 0

        # 最終イテレーションで記録
        if i == num_iter - 1:
            current_arr[:, t] = c2
            voltage_arr[:, t] = neurons_2.v_
            error_arr[:, t] = e2
            lambda_arr[:, :, t] = lambda2

    cost_arr[i] = cost
    print("\n cost:", cost)


# ==============================================================
# 結果の可視化
# --------------------------------------------------------------
# 6段構成のグラフで可視化：
#   (1) 出力膜電位
#   (2) 近似勾配（surrogate derivative）
#   (3) 誤差信号
#   (4) エリジビリティトレース
#   (5) 入力電流
#   (6) 入力スパイク列
# ==============================================================


def hide_ticks():
    """プロットの上・右軸を非表示"""
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")


t = np.arange(nt) * dt * 1e3
plt.figure(figsize=(8, 10))

plt.subplot(6, 1, 1)
plt.plot(t, voltage_arr[0], color="k")
plt.ylabel("Membrane\npotential (mV)")
hide_ticks()

plt.subplot(6, 1, 2)
plt.plot(t, et1.surrogate_derivative_fastsigmoid(u=voltage_arr[0]), color="k")
plt.ylabel("Surrogate\nderivative")
hide_ticks()

plt.subplot(6, 1, 3)
plt.plot(t, error_arr[0], color="k")
plt.ylabel("Error")
hide_ticks()

plt.subplot(6, 1, 4)
plt.plot(t, lambda_arr[0, 0], color="k")
plt.ylabel("$\lambda$ (eligibility)")
hide_ticks()

plt.subplot(6, 1, 5)
plt.plot(t, current_arr[0], color="k")
plt.ylabel("Input\ncurrent (pA)")
hide_ticks()

plt.subplot(6, 1, 6)
for i in range(N_in):
    plt.plot(t, x[:, i] * (i + 1), "ko", markersize=2, rasterized=True)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.xlim(0, t.max())
plt.ylim(0.5, N_in + 0.5)
hide_ticks()
plt.tight_layout()
plt.savefig("super_spike.svg")
plt.show()

# 学習コストの変化
plt.figure(figsize=(4, 3))
plt.plot(cost_arr, color="k")
plt.title("Cost over iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
hide_ticks()
plt.tight_layout()
plt.savefig("super_spike_cost.svg")
plt.show()
