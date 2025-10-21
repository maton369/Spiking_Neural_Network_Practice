# -*- coding: utf-8 -*-
"""
PyTorch版 Diehl & Cook (2015) スパイクニューラルネットワーク (SNN)
------------------------------------------------------------------
本スクリプトは、元の Chainer 実装を PyTorch に置き換えたものである。
STDP（Spike-Timing Dependent Plasticity）に基づく教師なし学習により、
MNISTデータセットを分類するスパイクニューラルネットワークを構築・訓練する。

参照：
Diehl & Cook, "Unsupervised learning of digit recognition using
spike-timing-dependent plasticity", Front. Comput. Neurosci. 9:99 (2015)
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 自作モジュールのインポート ---
from Neurons import ConductanceBasedLIF, DiehlAndCook2015LIF
from Synapses import SingleExponentialSynapse
from Connections import FullConnection, DelayConnection

np.random.seed(0)
torch.manual_seed(0)


#################
####  Utils  ####
#################
# ----------------------------------------------------------
# 画像をポアソンスパイク列に変換
# 入力画像（0～1の輝度値）をピクセルごとにスパイク確率に変換し、
# 時系列的にPoisson発火を生成する。
# ----------------------------------------------------------
def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32, norm=140):
    img, _ = dataset[i]
    img = img.view(-1).numpy()  # 28x28 → 784次元ベクトル化
    fr_tmp = max_fr * norm / np.sum(img)  # 平均発火率を画素値でスケール
    fr = fr_tmp * np.repeat(np.expand_dims(img, axis=0), n_time, axis=0)
    # ポアソン発火生成
    input_spikes = np.where(np.random.rand(n_time, 784) < fr * dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)
    return input_spikes


# ----------------------------------------------------------
# ニューロンをラベルに割り当てる関数
# 各ニューロンが最も多く発火したクラスをそのニューロンのラベルとする。
# ----------------------------------------------------------
def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
    n_neurons = spikes.shape[1]
    if rates is None:
        rates = np.zeros((n_neurons, n_labels)).astype(np.float32)

    for i in range(n_labels):
        n_labeled = np.sum(labels == i)
        if n_labeled > 0:
            indices = np.where(labels == i)[0]
            # 移動平均を用いたスパイク率の更新
            rates[:, i] = (
                alpha * rates[:, i] + np.sum(spikes[indices], axis=0) / n_labeled
            )

    # 各ニューロンの全体発火率で正規化
    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate == 0] = 1
    proportions = rates / np.expand_dims(sum_rate, 1)
    proportions[np.isnan(proportions)] = 0
    # 最大値のクラスを割り当て
    assignments = np.argmax(proportions, axis=1).astype(np.uint8)
    return assignments, proportions, rates


# ----------------------------------------------------------
# 割り当てたラベルを用いて、入力スパイク列を分類
# ----------------------------------------------------------
def prediction(spikes, assignments, n_labels):
    n_samples = spikes.shape[0]
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    for i in range(n_labels):
        n_assigns = np.sum(assignments == i)
        if n_assigns > 0:
            indices = np.where(assignments == i)[0]
            rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns
    return np.argmax(rates, axis=1).astype(np.uint8)


#################
####  Model  ####
#################
class DiehlAndCook2015Network:
    """
    Diehl & Cook (2015) モデルのPyTorch実装版。
    入力層 → 興奮性層 → 抑制性層 という3層構造を持ち、
    入力→E層の結合にSTDP学習を適用する。
    """

    def __init__(
        self,
        n_in=784,
        n_neurons=100,
        wexc=2.25,
        winh=0.875,
        dt=1e-3,
        wmin=0.0,
        wmax=5e-2,
        lr=(1e-2, 1e-4),
        update_nt=100,
    ):

        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax, self.wmin = wmax, wmin

        # --- ニューロン層 ---
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt)  # 興奮性層
        self.inh_neurons = ConductanceBasedLIF(n_neurons, dt=dt)  # 抑制性層

        # --- シナプス層 ---
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)
        self.input_trace = SingleExponentialSynapse(n_in, dt=dt, td=2e-2)
        self.exc_trace = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-2)

        # --- 接続構造 ---
        initW = 1e-3 * np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons, initW=initW)
        self.exc2inh_W = wexc * np.eye(n_neurons)  # 興奮性→抑制性
        self.inh2exc_W = (winh / (n_neurons - 1)) * (
            np.ones((n_neurons, n_neurons)) - np.eye(n_neurons)
        )
        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)

        # --- 内部変数の初期化 ---
        self.norm = 0.1
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.update_nt = update_nt
        self.n_neurons, self.n_in = n_neurons, n_in
        self.reset_trace()

    def reset_trace(self):
        """スパイク履歴とトレースのリセット"""
        self.s_in_ = np.zeros((self.update_nt, self.n_in))
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in))
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0

    def initialize_states(self):
        """ニューロン・シナプス・遅延構造の初期化"""
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()

    def __call__(self, s_in, stdp=True):
        """順伝播およびSTDP学習処理"""
        # --- 入力層 ---
        c_in = self.input_synapse(s_in)
        x_in = self.input_trace(s_in)
        g_in = self.input_conn(c_in)

        # --- 興奮性層 ---
        s_exc = self.exc_neurons(self.delay_input(g_in), self.g_inh)
        c_exc = self.exc_synapse(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_trace(s_exc)

        # --- 抑制性層 ---
        s_inh = self.inh_neurons(self.delay_exc2inh(g_exc), 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        # --- STDP学習 ---
        if stdp:
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.x_in_[self.tcount] = x_in
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1

            if self.tcount == self.update_nt:
                W = np.copy(self.input_conn.W)
                # 出力ニューロンごとに正規化
                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum
                # STDP更新則
                dW = self.lr_p * (self.wmax - W) * np.dot(self.s_exc_, self.x_in_)
                dW -= self.lr_m * W * np.dot(self.x_exc_, self.s_in_)
                clipped_dW = np.clip(dW / self.update_nt, -1e-3, 1e-3)
                self.input_conn.W = np.clip(W + clipped_dW, self.wmin, self.wmax)
                self.reset_trace()
        return s_exc


#################
#### Training ####
#################
if __name__ == "__main__":
    # --- MNISTデータの読み込み（PyTorch版） ---
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    labels = np.array([train_dataset[i][1] for i in range(10000)])

    # --- 時間・構成パラメータ ---
    dt, t_inj, t_blank = 1e-3, 0.35, 0.15
    nt_inj, nt_blank = round(t_inj / dt), round(t_blank / dt)
    n_neurons, n_labels, n_epoch, n_train = 100, 10, 10, 10000
    update_nt = nt_inj

    # --- ネットワーク初期化 ---
    network = DiehlAndCook2015Network(
        n_in=784, n_neurons=n_neurons, update_nt=update_nt
    )
    network.initialize_states()
    spikes = np.zeros((n_train, n_neurons)).astype(np.uint8)
    accuracy_all = np.zeros(n_epoch)
    blank_input = np.zeros(784)
    init_max_fr = 32

    results_save_dir = "./LIF_WTA_STDP_MNIST_results/"
    os.makedirs(results_save_dir, exist_ok=True)

    #################
    ## Simulation  ##
    #################
    for epoch in range(n_epoch):
        for i in tqdm(range(n_train)):
            max_fr = init_max_fr
            while True:
                # --- 入力スパイク生成 ---
                input_spikes = online_load_and_encoding_dataset(
                    train_dataset, i, dt, nt_inj, max_fr
                )
                spike_list = []

                # --- 画像入力フェーズ ---
                for t in range(nt_inj):
                    s_exc = network(input_spikes[t], stdp=True)
                    spike_list.append(s_exc)

                spikes[i] = np.sum(np.array(spike_list), axis=0)

                # --- ブランク入力フェーズ ---
                for _ in range(nt_blank):
                    _ = network(blank_input, stdp=False)

                # スパイク数が閾値に満たない場合は再試行
                num_spikes_exc = np.sum(np.array(spike_list))
                if num_spikes_exc >= 5:
                    break
                else:
                    max_fr += 16

        # --- ニューロンラベルの割り当て ---
        if epoch == 0:
            assignments, proportions, rates = assign_labels(spikes, labels, n_labels)
        else:
            assignments, proportions, rates = assign_labels(
                spikes, labels, n_labels, rates
            )

        # --- ラベル予測と精度評価 ---
        predicted_labels = prediction(spikes, assignments, n_labels)
        accuracy = np.mean(labels == predicted_labels)
        accuracy_all[epoch] = accuracy
        print(f"Epoch {epoch}: Accuracy = {accuracy*100:.2f}%")

        # --- 学習率減衰 ---
        network.lr_p *= 0.75
        network.lr_m *= 0.75

        # --- 重み保存 ---
        np.save(
            os.path.join(results_save_dir, f"weight_epoch{epoch}.npy"),
            network.input_conn.W,
        )

    #################
    ### Results ###
    #################
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(1, n_epoch + 1), accuracy_all * 100, color="k")
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_save_dir, "accuracy.svg"))

    # --- モデルパラメータ保存 ---
    np.save(os.path.join(results_save_dir, "assignments.npy"), assignments)
    np.save(os.path.join(results_save_dir, "weight.npy"), network.input_conn.W)
    np.save(
        os.path.join(results_save_dir, "exc_neurons_theta.npy"),
        network.exc_neurons.theta,
    )
