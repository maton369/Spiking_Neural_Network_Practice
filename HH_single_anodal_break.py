# -*- coding: utf-8 -*-
# ==============================================================
# Hodgkin-HuxleyニューロンモデルのPython実装
# --------------------------------------------------------------
# 本コードは、生物学的ニューロンの電気的活動（膜電位変化）を
# Na+, K+, および漏れ電流の3要素で再現するもの。
# 各チャネルの開閉確率は電位依存的に変化し、スパイク発火を生じる。
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class HodgkinHuxleyModel:
    def __init__(self, dt=1e-3, solver="RK4"):
        # -----------------------------
        # 生理学的パラメータ
        # -----------------------------
        self.C_m = 1.0  # 膜容量 [uF/cm^2]
        self.g_Na = 120.0  # Na+ の最大コンダクタンス [mS/cm^2]
        self.g_K = 36.0  # K+ の最大コンダクタンス [mS/cm^2]
        self.g_L = 0.3  # 漏れ電流のコンダクタンス [mS/cm^2]
        self.E_Na = 50.0  # Na+ の平衡電位 [mV]
        self.E_K = -77.0  # K+ の平衡電位 [mV]
        self.E_L = -54.387  # 漏れ電流の平衡電位 [mV]

        # 数値積分法設定
        self.solver = solver  # 使用する数値解法（"Euler" or "RK4"）
        self.dt = dt  # 時間刻み幅 [ms]

        # 状態変数 [V, m, h, n]
        # V: 膜電位, m/h/n: ゲート変数（チャネル開閉確率）
        self.states = np.array([-65, 0.05, 0.6, 0.32])
        self.I_m = None  # 外部入力電流 [μA/cm^2]

    # ==============================================================
    # 数値積分関数：Runge-Kutta4次法またはEuler法
    # ==============================================================
    def Solvers(self, func, x, dt):
        if self.solver == "RK4":
            # 高精度な4次Runge-Kutta法で1ステップ更新
            k1 = dt * func(x)
            k2 = dt * func(x + 0.5 * k1)
            k3 = dt * func(x + 0.5 * k2)
            k4 = dt * func(x + k3)
            return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        elif self.solver == "Euler":
            # 単純な陽的Euler法（誤差は大きいが高速）
            return x + dt * func(x)
        else:
            return None

    # ==============================================================
    # イオンチャネルのゲート関数 α(V), β(V)
    # --------------------------------------------------------------
    # これらの関数は、電位依存的にチャネル開閉速度を変化させる。
    # ==============================================================
    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80.0)

    # ==============================================================
    # 各イオン電流の計算
    # --------------------------------------------------------------
    # オームの法則に基づき、I_ion = g_ion * (V - E_ion)
    # Na+ と K+ はゲート変数による開閉確率を考慮。
    # ==============================================================
    def I_Na(self, V, m, h):
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        return self.g_K * n**4 * (V - self.E_K)

    def I_L(self, V):
        return self.g_L * (V - self.E_L)

    # ==============================================================
    # 微分方程式系（Hodgkin-Huxley方程式）
    # --------------------------------------------------------------
    # C_m * dV/dt = I_m - I_Na - I_K - I_L
    # ==============================================================
    def dALLdt(self, states):
        V, m, h, n = states

        # 膜電位の変化率：イオン電流のバランス
        dVdt = (self.I_m - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m

        # ゲート変数の変化率：電位依存的な確率変化
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        # [dV/dt, dm/dt, dh/dt, dn/dt] を返す
        return np.array([dVdt, dmdt, dhdt, dndt])

    # ==============================================================
    # モデルを1ステップ進める
    # --------------------------------------------------------------
    # 与えられた外部電流 I に対して状態を更新し、次の状態を返す。
    # ==============================================================
    def __call__(self, I):
        self.I_m = I
        states = self.Solvers(self.dALLdt, self.states, self.dt)
        self.states = states
        return states


##########
## Main ##
##########
dt = 0.01
T = 250  # 総シミュレーション時間 [ms]
nt = round(T / dt)  # 総ステップ数
time = np.arange(0.0, T, dt)  # 時間軸配列

# ==============================================================
# 外部刺激電流の定義 [μA/cm^2]
# --------------------------------------------------------------
# 時間ごとに電流の方向と大きさを変化させる。
# -10 → +10 → -20 → +20 と切り替え、発火応答を確認する。
# ==============================================================
I_inj = -10 * (time > 50) + 10 * (time > 100) - 20 * (time > 150) + 20 * (time > 200)

# モデルインスタンス作成（Euler法による積分）
HH_neuron = HodgkinHuxleyModel(dt=dt, solver="Euler")

# 結果記録用配列（V, m, h, n）
X_arr = np.zeros((nt, 4))

# ==============================================================
# 時間発展シミュレーション
# ==============================================================
for i in tqdm(range(nt)):
    X = HH_neuron(I_inj[i])  # 現在の入力電流で1ステップ進める
    X_arr[i] = X  # 状態を記録

# ==============================================================
# スパイク（発火）の検出
# --------------------------------------------------------------
# 膜電位が負→正へ変化したタイミングをスパイクとしてカウント。
# ==============================================================
spike = np.bitwise_and(X_arr[:-1, 0] < 0, X_arr[1:, 0] > 0)
print("Num. of spikes :", np.sum(spike))

# ==============================================================
# 結果の可視化
# --------------------------------------------------------------
# 上：膜電位の時間変化
# 中：外部電流の変化
# 下：ゲート変数 (m, h, n) の推移
# ==============================================================
plt.figure(figsize=(5, 5))

# 膜電位のプロット
plt.subplot(3, 1, 1)
plt.plot(time, X_arr[:, 0], color="k")
plt.ylabel("V (mV)")
plt.xlim(0, T)

# 外部刺激電流のプロット
plt.subplot(3, 1, 2)
plt.plot(time, I_inj, color="k")
plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
plt.xlim(0, T)
plt.ylim(-25, 10)
plt.tight_layout()

# ゲート変数の推移
plt.subplot(3, 1, 3)
plt.plot(time, X_arr[:, 1], "k", label="m")  # Na+ 活性化ゲート
plt.plot(time, X_arr[:, 2], "gray", label="h")  # Na+ 不活性化ゲート
plt.plot(time, X_arr[:, 3], "k", linestyle="dashed", label="n")  # K+ ゲート
plt.xlabel("t (ms)")
plt.ylabel("Gating Value")
plt.legend(loc="upper left")
plt.tight_layout()

# PDFとして保存し表示
plt.savefig("HH_model_anodal_break.pdf")
plt.show()
