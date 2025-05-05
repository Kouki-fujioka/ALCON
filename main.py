import sys
import csv
from dataclasses import dataclass
from typing import Dict, List
from enum import IntEnum
import cv2
import numpy as np
from typing import Dict, List
import tqdm
import collections
from utils import *

file_num = 1    # 1 <= file_num <= 5
CSV_FILE_PATH = f'./Problem_0{file_num}/input-Problem_0{file_num}.csv'  # 入力ファイル
OUTPUT_CSV_FILE_PATH = f'./output-Problem_0{file_num}.csv'  # 出力ファイル
INPUT_VIDEO_PATH = f'./Problem_0{file_num}/Problem_0{file_num}.mp4' # 入力動画
OUTPUT_VIDEO_PATH = f'./Out_Problem_0{file_num}.MP4'    # 出力動画

class FishNames(IntEnum):   # 魚の種類 (列挙型クラス)
    KAWAMEDAKA = 0  # name = KAWAMEDAKA, value = 0
    HIMEDAKA = 1    # name = HIMEDAKA, value = 1
    KOKIN = 2   # name = KOKIN, value = 2

    @staticmethod   # 静的メソッド
    def from_name(name):    # FishNames のメンバを検索し, 一致するメンバを返却するメソッド (列挙型メソッド)
        for v in FishNames: # FishNames クラスのメンバをループ
            if name == v.name:return v    # 一致するメンバを返却
        raise ValueError(f'{name} is not a valid FishNames!')   # 一致するメンバが見つからなかった場合, エラーメッセージを表示

class Config:   # Config クラス
    def __init__(self, folder_name, width, height, aspect_ratio_x, aspect_ratio_y, fps, fish_variety, input_video_path):    # コンストラクタ
        self.folder_name = folder_name  # 入学データのフォルダ名 (CSV ファイルに書き込む用)
        self.width = width  # 解像度 (幅)
        self.height = height    # 解像度 (高さ)
        self.aspect_ratio_x = aspect_ratio_x    # アスペクト比 (x)
        self.aspect_ratio_y = aspect_ratio_y    # アスペクト比 (y)
        self.fps = fps  # フレームレート
        self.fish_variety = fish_variety    # 魚の総種類数
        self.input_video_path = input_video_path    # 入力動画のパス

    @staticmethod   # 静的メソッド
    def from_csv(path): # CSV ファイルから Config クラスのインスタンスを生成するメソッド
        with open(path) as f:   # ファイルオープン
            reader = csv.reader(f)  # CSV ファイルの読み込み
            lines = list(map(lambda e:e[0], reader))    # CSV ファイルの各行を読み込み, リストを作成
        return Config(lines[0], int(lines[1]), int(lines[2]), int(lines[3]), int(lines[4]), float(lines[5]), int(lines[6]), lines[7])   # Config クラスのインスタンスを返却

@dataclass  # データクラス (自動的に初期化)
class Rect: # 矩形領域を表現するクラス
    x:int   # x 座標 (type hint:int 型)
    y:int   # y 座標 (type hint:int 型)
    w:int   # 幅 (type hint:int 型)
    h:int   # 高さ (type hint:int 型)
"""
def __init__(self, x: int, y: int, w: int, h: int):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
"""

def rect_color(fish_name:FishNames):    # 魚の種類に基づいて, 矩形領域の枠線の色を返却するメソッド (type hint:FishNames 型)
    if fish_name == FishNames.KAWAMEDAKA:   # KAWAMEDAKA の場合
        return 0, 100, 0    # 枠線の色 (BGR) を返却
    elif fish_name == FishNames.HIMEDAKA:   # HIMEDAKA の場合
        return 100, 0, 0    # 枠線の色 (BGR) を返却
    elif fish_name == FishNames.KOKIN:  # KOKIN の場合
        return 0, 0, 100    # 枠線の色 (BGR) を返却
    else:   # その他の場合
        return 100, 100, 100    # 枠線の色 (BGR) を返却

def detect_on_frame(frame, background_subtractor): # 1フレーム内に含まれる魚を検出し, 魚の矩形領域を取得するメソッド
    detections:Dict[FishNames, List[Rect]] = {
        FishNames.KAWAMEDAKA:[],    # 検出された魚 (KAWAMEDAKA) の矩形領域を格納するリストを初期化
        FishNames.HIMEDAKA:[],  # 検出された魚 (HIMEDAKA) の矩形領域を格納するリストを初期化
        FishNames.KOKIN:[]  # 検出された魚 (KOKIN) の矩形領域を格納するリストを初期化
    }   # キー:FishNames, 値:検出された魚の種類と矩形領域を格納するリスト

    yuv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)    # YUV 色空間に変換
    yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])   # 輝度チャネルに対してヒストグラム平坦化
    frame = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)    # BGR に再変換

    fg_mask = background_subtractor.apply(frame)    # 前景マスクの生成

    "KOKIN"
    red_area = apply_hsv_threshold(frame, np.array([0, 240, 50]), np.array([5, 255, 150]))  # HSV 色空間において指定した色域 (赤色) に該当するピクセルを白 (255), 背景を黒 (0) で表現する2値画像を返却
    """
    frame:処理対象の画像 (1フレーム)
    np.array([0, 240, 50]):検出する色の下限値 (色相, 彩度, 明度)
    np.array([5, 255, 150]):検出する色の上限値 (色相, 彩度, 明度)
    """

    combined_mask = cv2.bitwise_and(fg_mask, red_area)  # 背景差分法を適用

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)  # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    denoised = np.zeros_like(red_area)  # red_area の全ピクセルを0 (黒) で初期化した2値画像を取得
    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        if 2000 <= area <= 5000: # 面積が2000以上5000以下の場合
            denoised[labels == label_idx] = 255 # 面積が2000以上5000以下の領域を denoised 画像に追加し, 領域を白 (255) に設定
    dilated = cv2.dilate(denoised, (3, 3), iterations = 20) # denoised 画像の白領域を (3, 3) の正方形カーネルを用いて膨張 (20回)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)   # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        detections[FishNames.KOKIN].append(Rect(x, y, w, h))    # KOKIN の矩形領域をリストに追加

    "HIMEDAKA"
    scarlet_area = apply_hsv_threshold(frame, np.array([8, 230, 50]), np.array([21, 255, 150])) # HSV 色空間において指定した色域 (緋色) に該当するピクセルを白 (255), 背景を黒 (0) で表現する2値画像を返却
    """
    frame:処理対象の画像 (1フレーム)
    np.array([8, 230, 50]):検出する色の下限値 (色相, 彩度, 明度)
    np.array([21, 255, 150]):検出する色の上限値 (色相, 彩度, 明度)
    """

    combined_mask = cv2.bitwise_and(fg_mask, scarlet_area)  # 背景差分法を適用

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)  # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    denoised = np.zeros_like(scarlet_area)  # scarlet_area の全ピクセルを0 (黒) で初期化した2値画像を取得
    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        if 400 <= area <= 500: # 面積が400以上500以下の場合
            denoised[labels == label_idx] = 255 # 面積が400以上500以下の領域を denoised 画像に追加し, 領域を白 (255) に設定
    dilated = cv2.dilate(denoised, (3, 3), iterations = 5) # denoised 画像の白領域を (3, 3) の正方形カーネルを用いて膨張 (10回)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)   # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        detections[FishNames.HIMEDAKA].append(Rect(x, y, w, h)) # HIMEDAKA の矩形領域をリストに追加

    "KAWAMEDAKA"
    black_area = apply_hsv_threshold(frame, np.array([5, 230, 40]), np.array([21, 255, 100]))   # HSV 色空間において指定した色域 (黒色) に該当するピクセルを白 (255), 背景を黒 (0) で表現する2値画像を返却
    """
    frame:処理対象の画像 (1フレーム)
    np.array([5, 230, 40]):検出する色の下限値 (色相, 彩度, 明度)
    np.array([21, 255, 100]):検出する色の上限値 (色相, 彩度, 明度)
    """

    combined_mask = cv2.bitwise_and(fg_mask, black_area)    # 背景差分法を適用

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask) # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    denoised = np.zeros_like(black_area)    # black_area の全ピクセルを0 (黒) で初期化した2値画像を取得
    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        if 400 <= area <= 500: # 面積が400以上500以下の場合
            denoised[labels == label_idx] = 255 # 面積が400以上500以下の領域を denoised 画像に追加し, 領域を白 (255) に設定
    dilated = cv2.dilate(denoised, (3, 3), iterations = 20) # denoised 画像の白領域を (3, 3) の正方形カーネルを用いて膨張 (20回)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)   # ラベリング処理を行い, 画像中の白領域を識別
    """
    nlabels:検出されたラベル数 (背景 (1) + 白領域の数 (n))
    labels:各ピクセルがどのラベル (背景, 白領域) に属するかを示す2次元配列
    stats:各ラベル (背景, 白領域) の統計情報 (x, y, width, height, area) を格納する配列
    centroids:各ラベル (背景, 白領域) の重心座標を格納する配列
    """

    for label_idx in range(1, nlabels): # 1 <= label_idx <= nlabels - 1 (背景ラベル (0) はスキップ)
        x, y, w, h, area = stats[label_idx] # 各ラベル (白領域) の統計情報を取得
        detections[FishNames.KAWAMEDAKA].append(Rect(x, y, w, h))   # KAWAMEDAKA の矩形領域をリストに追加

    return detections   # 1フレーム内で検出された魚の種類と矩形領域を返却

def main(): # メイン処理
    config = Config.from_csv(CSV_FILE_PATH) # CSV ファイルの読み込み (Config クラスのインスタンスを生成)
    video_reader = cv2.VideoCapture(INPUT_VIDEO_PATH)   # 動画の読み込み
    if not video_reader.isOpened(): # 動画が開けない場合
        print('入力動画ファイルを開けません！') # エラーメッセージを表示
        sys.exit(1) # プログラム終了

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # 出力用動画ファイルのコーデックを指定 (MP4)
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, config.fps, (config.width, config.height))    # 出力用動画ファイルの設定
    fish_count:Dict[FishNames, List[int]] = {}  # 各魚の検出履歴を保持する辞書を初期化
    for fish_name in FishNames: # FishNames クラスのメンバをループ
        fish_count[fish_name] = []  # 各魚の検出履歴のリストを初期化

    progress_bar = tqdm.tqdm(total = video_reader.get(cv2.CAP_PROP_FRAME_COUNT))    # 動画の処理進捗をプログレスバーで表示
    background_subtractor = cv2.createBackgroundSubtractorMOG2()    # 背景差分法の初期化
    frame_count = 0 # フレーム数
    while video_reader.isOpened():  # 動画の読み込みに成功した場合, 無限ループ
        progress_bar.update(1)  # プログレスバーを更新
        bool, frame = video_reader.read()   # 1フレームずつ読み込み (bool = True)
        frame_count += 1    # インクリメント
        if not bool:    # 読み込みに失敗, 読み込みが完了した場合
            break   # ループ終了

        detections = detect_on_frame(frame, background_subtractor) # 1フレーム内に含まれる魚を検出し, 魚の矩形領域を取得
        result_frame = np.copy(frame)   # 1フレームずつコピー (矩形領域とテキスト描画, 結果出力用)
        for fish_name, rects in detections.items(): # 1フレーム内で検出された魚の種類と矩形領域をループ
            count_of_this_fish = len(rects) # 1フレーム内で検出された魚の数
            fish_count[fish_name].append(count_of_this_fish)    # 各魚の検出数をリストに追加
            for i, rect in enumerate(rects):    # 矩形領域オブジェクトをループ (i は矩形領域の番号 (0 ~ 矩形領域の数 - 1))
                cv2.rectangle(result_frame, # 矩形領域を描画する1フレーム
                              pt1 = (rect.x, rect.y),   # 矩形領域の座標 (左上)
                              pt2 = (rect.x + rect.w, rect.y + rect.h), # 矩形領域の座標 (右下)
                              color = rect_color(fish_name),    # 矩形領域の枠線の色
                              thickness = 2)    # 矩形領域の枠線の太さ
                cv2.putText(result_frame,   # テキストを描画する1フレーム
                            f'{fish_name.name}:#{i + 1}',   # 魚の種類と矩形領域の番号 (1 ~ 矩形領域の数)
                            (rect.x, rect.y - 10),  # テキストを表示する座標
                            cv2.FONT_HERSHEY_PLAIN, # フォントの種類
                            1,  # フォントサイズ
                            (255, 0, 0),    # テキストの色
                            1,  # テキストの線の太さ
                            cv2.LINE_AA)    # ラインの種類

        video_writer.write(result_frame)    # 結果出力用フレームを出力用動画ファイルに書き込み

    with open(OUTPUT_CSV_FILE_PATH, 'w', newline='') as f:  # ファイルオープン
        writer = csv.writer(f)  # ライターオブジェクトを作成
        writer.writerow([config.folder_name])   # フォルダ名を書き込み
        writer.writerow([config.fish_variety])  # 魚の総種類数を書き込み
        for fish_name, counts in fish_count.items():    # 各魚の検出履歴をループ
            count_sum = sum(counts) # 検出数の総和を取得
            count_ave = int(np.ceil(count_sum / frame_count))   # 検出数の平均を取得
            writer.writerow([int(fish_name), count_ave])    #  検出された魚の種類 (value) と平均検出数を書き込み

if __name__ == '__main__':  # スクリプトが直接実行された場合
    main()  # メイン処理を実行
