# -*- coding: utf-8 -*-
"""
学習済みSNNの重み可視化スクリプト
------------------------------------
Diehl & Cook (2015) モデルに基づいて学習された
入力層 → 興奮性ニューロン層の結合重みを
28×28の画像としてプロットし、学習された特徴マップを可視化する。

【目的】
- 各ニューロンがどのような入力パターン（数字・特徴）に反応するかを確認
- STDPにより形成された空間的フィルタ（受容野）を視覚的に理解する
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 可視化設定パラメータ
# -----------------------------
epoch = 29  # 可視化対象とする学習エポック
n_neurons = 100  # 興奮性ニューロン数（Diehl & Cookモデルでは通常100）

# -----------------------------
# 結果ファイルの読み込み
# -----------------------------
results_save_dir = "./LIF_WTA_STDP_MNIST_results/"  # 学習結果の保存ディレクトリ

# 指定エポックの学習済み重みをロード
# 形状: (n_neurons, n_inputs)
input_conn_W = np.load(results_save_dir + "weight_epoch" + str(epoch) + ".npy")

# 各ニューロンの受容野を28x28にリシェイプ（MNIST画像サイズと対応）
reshaped_W = np.reshape(input_conn_W, (n_neurons, 28, 28))

# -----------------------------
# 可視化設定
# -----------------------------
fig = plt.figure(figsize=(6, 6))
# サブプロット間の余白を調整（詰めてグリッド状に表示）
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# サブプロットの行・列数（√N×√Nに整列）
row = col = int(np.sqrt(n_neurons))

# -----------------------------
# 各ニューロンの重みを画像化
# -----------------------------
for i in tqdm(range(n_neurons), desc="Plotting receptive fields"):
    ax = fig.add_subplot(row, col, i + 1, xticks=[], yticks=[])
    # 重みをグレースケールで描画
    ax.imshow(reshaped_W[i], cmap="gray")

# -----------------------------
# 出力の保存と表示
# -----------------------------
# エポック番号付きで保存（例: weights_29.png）
plt.savefig("weights_" + str(epoch) + ".png")

# 画面上に描画
plt.show()

"""
🧠 出力概要：
- 各サブプロットは1つの興奮性ニューロンの受容野（重み分布）を表す
- 明るい領域：ニューロンが強く反応するピクセル（興奮性）
- 暗い領域：反応が弱いピクセル（抑制性）
この分布を見ることで、どの数字・形状・ストロークに反応するかが直感的に理解できる。
"""
