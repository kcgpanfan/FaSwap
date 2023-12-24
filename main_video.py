# 導入所需的模塊
import os
import cv2
import numpy
import argparse
import streamlit as st
from PIL import Image
from moviepy.editor import VideoFileClip
from face_detection import select_face, select_all_faces
from face_swap import face_swap

# 定義一個裝飾器，用於緩存人臉交換的結果
@st.cache(allow_output_mutation=True)
def face_swap_cached(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args):
    return face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args)

# 創建網頁的標題
st.title('Face Swap App')

# 創建文件上傳的選項
uploaded_source_file = st.file_uploader("Source File", type=['jpg', 'png', 'jpeg'])
uploaded_target_file = st.file_uploader("Target File", type=['mp4', 'avi', 'mov'])

# 創建一個佔位符，用於顯示源圖片
source_image_placeholder = st.empty()

# 創建一個佔位符，用於顯示目標影片
target_video_placeholder = st.empty()

# 創建一個佔位符，用於顯示結果影片
result_video_placeholder = st.empty()

# 創建一個按鈕，用於開始人臉交換
start_button = st.button('Start Face Swap')

# 如果上傳了源文件和目標文件，則進行以下操作
if uploaded_source_file is not None and uploaded_target_file is not None:
    # 將源文件轉換為PIL圖片
    source_image = Image.open(uploaded_source_file)

    # 在佔位符中顯示源圖片
    source_image_placeholder.write('Source Image:')
    source_image_placeholder.image(source_image, channels="RGB")

    # 讀取目標影片，並保留音軌
    target_video = VideoFileClip(uploaded_target_file.name)
    audio = target_video.audio

    # 在佔位符中顯示目標影片
    target_video_placeholder.write('Target Video:')
    target_video_placeholder.video(target_video)

    # 如果按下了按鈕，則進行以下操作
    if start_button:
        # 創建一個參數對象，用於傳遞給人臉交換的函數
        args = argparse.Namespace(
            correct_color=True,
            warp_2d=False
        )

        # 選擇源圖片中的人臉
        src_img = numpy.array(source_image)
        src_points, src_shape, src_face = select_face(src_img)

        # 定義一個函數，用於對每一幀進行人臉交換
        def make_frame(t):
            # 獲取當前幀的圖片
            frame = target_video.get_frame(t)

            # 選擇當前幀中的所有人臉
            dst_faceBoxes = select_all_faces(frame)

            # 如果沒有檢測到人臉，則返回原始幀
            if src_points is None or dst_faceBoxes is None:
                return frame

            # 對每個目標人臉進行人臉交換，並緩存結果
            output = frame
            for k, dst_face in dst_faceBoxes.items():
                output = face_swap_cached(src_face, dst_face["face"], src_points,
                                          dst_face["points"], dst_face["shape"],
                                          output, args)

            return output

        # 生成結果影片
        result_video = target_video.fl(make_frame)

        # 將結果影片與音軌合併
        result_video = result_video.set_audio(audio)

        # 顯示結果影片
        result_video_placeholder.write('Result Video:')
        result_video_placeholder.video(result_video, format='webm')
