# -*- coding: utf-8 -*-
"""
PyTorch版 Diehl & Cook (2015) SNNモデルのテストスクリプト
----------------------------------------------------
本スクリプトは、学習済みのSTDPネットワークを用いて
MNISTテストデータを分類し、最終的な分類精度を評価します。

【概要】
- 入力画像をポアソンスパイク列に変換
- ネットワークへ刺激を与え、スパイク応答を観測
- 学習済みラベル割り当てを用いて予測を生成
- 精度(accuracy)を算出して出力

※ Chainer依存を完全に排除し、PyTorch版MNISTデータローダを使用。
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 学習済みネットワークの構成要素をインポート
from LIF_WTA_STDP_MNIST import online_load_and_encoding_dataset, prediction
from LIF_WTA_STDP_MNIST import DiehlAndCook2015Network

#################
####  Main   ####
#################

# -----------------------------
# シミュレーションパラメータ設定
# -----------------------------
dt = 1e-3  # タイムステップ [s]
t_inj = 0.350  # 刺激入力時間 [s]
t_blank = 0.150  # 入力なしのリセット時間 [s]
nt_inj = round(t_inj / dt)  # 刺激時間のステップ数
nt_blank = round(t_blank / dt)  # ブランク時間のステップ数

n_neurons = 100  # 興奮性/抑制性ニューロンの数
n_labels = 10  # MNISTのラベル数 (0〜9)
n_test = 1000  # テストデータに使用するサンプル数
update_nt = nt_inj  # STDP更新間隔（推論では不要だが構造上必要）

# -----------------------------
# PyTorchでMNISTデータセットを読み込み
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# MNISTのラベルをNumPy配列に変換
labels = np.array([test_dataset[i][1] for i in range(n_test)])

# -----------------------------
# ネットワーク構築とパラメータ読込
# -----------------------------
results_save_dir = "./LIF_WTA_STDP_MNIST_results/"  # 学習済みモデルのディレクトリ

# Diehl & Cook (2015) ネットワーク構築
network = DiehlAndCook2015Network(
    n_in=784, n_neurons=n_neurons, wexc=2.25, winh=0.85, dt=dt
)

# ニューロン・シナプス状態の初期化
network.initialize_states()

# 学習済みパラメータのロード
network.input_conn.W = np.load(results_save_dir + "weight.npy")
network.exc_neurons.theta = np.load(results_save_dir + "exc_neurons_theta.npy")

# 推論時は閾値上昇を無効化
network.exc_neurons.theta_plus = 0

# -----------------------------
# 評価用変数の初期化
# -----------------------------
spikes = np.zeros((n_test, n_neurons)).astype(np.uint8)  # 各サンプルの発火回数を保存
blank_input = np.zeros(784)  # 入力なし状態（リセット用）
init_max_fr = 32  # 初期ポアソン発火率

#################
## Simulation  ##
#################
# -------------------------------------------------
# 各テストサンプルに対してスパイク応答を取得
# -------------------------------------------------
for i in tqdm(range(n_test), desc="Testing"):
    max_fr = init_max_fr  # 入力スパイク率の初期値

    while True:
        # (1) 画像をポアソンスパイク列に変換
        input_spikes = online_load_and_encoding_dataset(
            test_dataset, i, dt, nt_inj, max_fr
        )

        spike_list = []  # 各サンプルのスパイク履歴を保持

        # (2) 画像刺激をネットワークに入力
        for t in range(nt_inj):
            s_exc = network(input_spikes[t], stdp=False)  # STDP無効（推論モード）
            spike_list.append(s_exc)

        # (3) 発火回数を集計
        spikes[i] = np.sum(np.array(spike_list), axis=0)

        # (4) ブランク期間（刺激なしで膜電位をリセット）
        for _ in range(nt_blank):
            _ = network(blank_input, stdp=False)

        # (5) スパイク数が閾値以上なら次のサンプルへ
        num_spikes_exc = np.sum(np.array(spike_list))
        if num_spikes_exc >= 5:
            break  # 充分にスパイクが発生した場合
        else:
            max_fr += 16  # 発火が少ない場合、入力強度を増加

# -----------------------------
# 分類予測および精度評価
# -----------------------------
# 学習時に保存されたニューロンとラベルの対応を読み込み
assignments = np.load(results_save_dir + "assignments.npy")

# 発火パターンに基づき予測ラベルを算出
predicted_labels = prediction(spikes, assignments, n_labels)

# 正答率（Accuracy）の計算
accuracy = np.mean(labels == predicted_labels).astype(np.float16)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 結果の保存（任意）
# -----------------------------
os.makedirs(results_save_dir, exist_ok=True)
np.save(results_save_dir + "test_spikes.npy", spikes)
np.save(results_save_dir + "test_predictions.npy", predicted_labels)
print(f"Results saved to: {results_save_dir}")
