import cv2
import numpy as np


def posterize(image, levels):
    # ポスタリゼーション処理のために, 各チャンネルの色階調を計算
    r_levels = np.linspace(0, 255, levels + 1).astype(np.uint8)
    g_levels = np.linspace(0, 255, levels + 1).astype(np.uint8)
    b_levels = np.linspace(0, 255, levels + 1).astype(np.uint8)

    # ポスタリゼーションを適用
    r_indices = np.digitize(image[:, :, 0], r_levels) - 1
    g_indices = np.digitize(image[:, :, 1], g_levels) - 1
    b_indices = np.digitize(image[:, :, 2], b_levels) - 1

    # 各チャンネルに対して色階調を適用
    posterized_image = np.zeros_like(image)
    posterized_image[:, :, 0] = r_levels[r_indices]
    posterized_image[:, :, 1] = g_levels[g_indices]
    posterized_image[:, :, 2] = b_levels[b_indices]

    return posterized_image


def apply_hsv_threshold(image, lower_hsv, upper_hsv):
    # BGR 画像を HSV 色空間に変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 色空間での閾値処理
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # 閾値処理を適用して抽出した画像を取得
    # thresholded_image = cv2.bitwise_and(image, image, mask = mask)

    # return thresholded_image
    return mask
