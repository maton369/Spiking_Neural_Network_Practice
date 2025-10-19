# -*- coding: utf-8 -*-
# ==============================================================
# スパイキングニューロンモデル群の実装
# --------------------------------------------------------------
# このスクリプトでは、4種類の代表的なスパイキングニューロンモデルを定義している。
# それぞれ異なる生理学的仮定とダイナミクスを持ち、神経回路の多様な発火挙動を再現可能。
# ==============================================================

import numpy as np


# ==============================================================
# 1️⃣ Current-Based LIF (Leaky Integrate-and-Fire) モデル
# --------------------------------------------------------------
# 電流ベースの単純なLIFモデル。入力電流Iに比例して膜電位が上昇し、
# 閾値を超えるとスパイクを発生、その後リセットされる。
# ==============================================================
class CurrentBasedLIF:
    def __init__(
        self,
        N,
        dt=1e-4,
        tref=5e-3,
        tc_m=1e-2,
        vrest=-60,
        vreset=-60,
        vthr=-50,
        vpeak=20,
    ):
        """
        Current-based Leaky integrate-and-fire model.
        """
        # モデルパラメータの設定
        self.N = N  # ニューロン数
        self.dt = dt  # 時間刻み幅 [s]
        self.tref = tref  # 不応期 [s]
        self.tc_m = tc_m  # 膜時定数 [s]
        self.vrest = vrest  # 静止膜電位 [mV]
        self.vreset = vreset  # リセット電位 [mV]
        self.vthr = vthr  # 発火閾値 [mV]
        self.vpeak = vpeak  # スパイクピーク電位 [mV]

        # 状態変数
        self.v = self.vreset * np.ones(N)  # 各ニューロンの初期電位
        self.v_ = None  # 発火時の膜電位記録
        self.tlast = 0  # 最後に発火した時刻
        self.tcount = 0  # シミュレーション経過ステップ数

    def initialize_states(self, random_state=False):
        # ニューロンの初期状態を設定
        if random_state:
            # ランダム初期化（vreset〜vthrの範囲）
            self.v = self.vreset + np.random.rand(self.N) * (self.vthr - self.vreset)
        else:
            # 一定値で初期化
            self.v = self.vreset * np.ones(self.N)
        self.tlast = 0
        self.tcount = 0

    def __call__(self, I):
        # 膜電位の変化率を計算（漏れ項 + 入力電流）
        dv = (self.vrest - self.v + I) / self.tc_m
        # 不応期経過後のみ膜電位を更新
        v = self.v + ((self.dt * self.tcount) > (self.tlast + self.tref)) * dv * self.dt

        # 発火判定（閾値を超えると1）
        s = 1 * (v >= self.vthr)

        # 発火時刻の更新
        self.tlast = self.tlast * (1 - s) + self.dt * self.tcount * s
        # 発火中は膜電位をピーク値にする
        v = v * (1 - s) + self.vpeak * s
        # 発火時の電位も記録（グラフ描画用）
        self.v_ = v
        # スパイク発生後はリセット電位に戻す
        self.v = v * (1 - s) + self.vreset * s
        # ステップ数を進める
        self.tcount += 1

        return s


# ==============================================================
# 2️⃣ Conductance-Based LIF モデル
# --------------------------------------------------------------
# シナプス入力を電流ではなくコンダクタンス（伝導率）として表す。
# 興奮性・抑制性入力を明示的に扱える、生理学的によりリアルなモデル。
# ==============================================================
class ConductanceBasedLIF:
    def __init__(
        self,
        N,
        dt=1e-4,
        tref=5e-3,
        tc_m=1e-2,
        vrest=-60,
        vreset=-60,
        vthr=-50,
        vpeak=20,
        e_exc=0,
        e_inh=-100,
    ):
        """
        Conductance-based Leaky integrate-and-fire model.
        """
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m
        self.vrest = vrest
        self.vreset = vreset
        self.vthr = vthr
        self.vpeak = vpeak

        # シナプス平衡電位（興奮性と抑制性）
        self.e_exc = e_exc
        self.e_inh = e_inh

        self.v = self.vreset * np.ones(N)
        self.v_ = None
        self.tlast = 0
        self.tcount = 0

    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N) * (self.vthr - self.vreset)
        else:
            self.v = self.vreset * np.ones(self.N)
        self.tlast = 0
        self.tcount = 0

    def __call__(self, g_exc, g_inh):
        # シナプス電流 = g * (E - V)
        I_synExc = g_exc * (self.e_exc - self.v)
        I_synInh = g_inh * (self.e_inh - self.v)
        # 膜電位の変化率
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m
        # 不応期判定付き更新
        v = self.v + ((self.dt * self.tcount) > (self.tlast + self.tref)) * dv * self.dt

        # 発火検出
        s = 1 * (v >= self.vthr)
        # 発火時刻を更新
        self.tlast = self.tlast * (1 - s) + self.dt * self.tcount * s
        # 発火中の膜電位をピーク電位へ
        v = v * (1 - s) + self.vpeak * s
        self.v_ = v
        # 発火後はリセット電位へ戻す
        self.v = v * (1 - s) + self.vreset * s
        self.tcount += 1

        return s


# ==============================================================
# 3️⃣ Diehl & Cook (2015) LIF モデル
# --------------------------------------------------------------
# LIFに動的閾値(theta)を導入し、発火頻度を自己調整する仕組みを持つ。
# SNNの教師なし学習（STDP）などでよく用いられる。
# ==============================================================
class DiehlAndCook2015LIF:
    def __init__(
        self,
        N,
        dt=1e-3,
        tref=5e-3,
        tc_m=1e-1,
        vrest=-65,
        vreset=-65,
        init_vthr=-52,
        vpeak=20,
        theta_plus=0.05,
        theta_max=35,
        tc_theta=1e4,
        e_exc=0,
        e_inh=-100,
    ):
        """
        Diehl & Cook (2015) Leaky integrate-and-fire model.
        """
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m
        self.vreset = vreset
        self.vrest = vrest
        self.init_vthr = init_vthr

        # 動的閾値パラメータ
        self.theta = np.zeros(N)
        self.theta_plus = theta_plus
        self.theta_max = theta_max
        self.tc_theta = tc_theta
        self.vpeak = vpeak

        # シナプス平衡電位
        self.e_exc = e_exc
        self.e_inh = e_inh

        # 状態初期化
        self.v = self.vreset * np.ones(N)
        self.vthr = self.init_vthr
        self.v_ = None
        self.tlast = 0
        self.tcount = 0

    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N) * (self.vthr - self.vreset)
        else:
            self.v = self.vreset * np.ones(self.N)
        self.vthr = self.init_vthr
        self.theta = np.zeros(self.N)
        self.tlast = 0
        self.tcount = 0

    def __call__(self, g_exc, g_inh):
        # 興奮性・抑制性シナプス電流
        I_synExc = g_exc * (self.e_exc - self.v)
        I_synInh = g_inh * (self.e_inh - self.v)
        # 電位変化率
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m
        # 不応期後に更新
        v = self.v + ((self.dt * self.tcount) > (self.tlast + self.tref)) * dv * self.dt

        # 発火判定
        s = 1 * (v >= self.vthr)
        # 動的閾値の更新（適応性）
        theta = (1 - self.dt / self.tc_theta) * self.theta + self.theta_plus * s
        # 上限を設定
        self.theta = np.clip(theta, 0, self.theta_max)
        self.vthr = self.theta + self.init_vthr

        # 発火時刻更新と膜電位処理
        self.tlast = self.tlast * (1 - s) + self.dt * self.tcount * s
        v = v * (1 - s) + self.vpeak * s
        self.v_ = v
        self.v = v * (1 - s) + self.vreset * s
        self.tcount += 1

        return s


# ==============================================================
# 4️⃣ Izhikevich モデル
# --------------------------------------------------------------
# 生物学的リアリズムと計算効率の両立を目指したモデル。
# Hodgkin-Huxleyモデルの振る舞いをわずか2変数(v, u)で近似。
# ==============================================================
class IzhikevichNeuron:
    def __init__(
        self,
        N,
        dt=0.5,
        C=250,
        a=0.01,
        b=-2,
        k=2.5,
        d=200,
        vrest=-60,
        vreset=-65,
        vthr=-20,
        vpeak=30,
    ):
        """
        Izhikevich neuron model.
        """
        # 定数パラメータ
        self.N = N
        self.dt = dt  # 時間刻み幅 [ms]
        self.C = C  # 膜容量 [pF]
        self.a = a  # 適応時定数
        self.b = b  # 共鳴パラメータ
        self.d = d  # 適応ジャンプ量
        self.k = k  # 電位増幅係数
        self.vrest = vrest
        self.vreset = vreset
        self.vthr = vthr
        self.vpeak = vpeak

        # 状態変数の初期化
        self.u = np.zeros(N)  # 適応電流
        self.v = self.vrest * np.ones(N)  # 膜電位
        self.v_ = self.v

    def initialize_states(self, random_state=False):
        if random_state:
            # 初期膜電位をランダム化
            self.v = self.vreset + np.random.rand(self.N) * (self.vthr - self.vreset)
        else:
            # 静止電位から開始
            self.v = self.vrest * np.ones(self.N)
        self.u = np.zeros(self.N)

    def __call__(self, I):
        # 膜電位と適応変数の時間発展を計算
        dv = (
            self.k * (self.v - self.vrest) * (self.v - self.vthr) - self.u + I
        ) / self.C
        v = self.v + self.dt * dv
        u = self.u + self.dt * (self.a * (self.b * (self.v_ - self.vrest) - self.u))

        # 発火検出
        s = 1 * (v >= self.vpeak)

        # 発火時の状態更新
        self.u = u + self.d * s
        self.v = v * (1 - s) + self.vreset * s
        self.v_ = self.v

        return s
