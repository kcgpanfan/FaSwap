#! /usr/bin/env python
# 導入所需的模塊
import os
import cv2
import logging
import argparse
import streamlit as st
import numpy as np
from PIL import Image

# 導入或定義人臉檢測和人臉交換的函數
from face_detection import select_face
from face_swap import face_swap

# 定義VideoHandler類，用於處理視頻中的人臉交換
class VideoHandler(object):
    def __init__(self, video_path=0, img_path=None, args=None):
        self.src_points, self.src_shape, self.src_face = select_face(cv2.imread(img_path))
        if self.src_points is None:
            st.write('No face detected in the source image !!!')
            exit(-1)
        self.args = args
        self.video = cv2.VideoCapture(video_path)
        self.writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                      self.video.get(cv2.CAP_PROP_FPS),
                                      (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def start(self):
        while self.video.isOpened():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            _, dst_img = self.video.read()
            dst_points, dst_shape, dst_face = select_face(dst_img, choose=False)
            if dst_points is not None:
                dst_img = face_swap(self.src_face, dst_face, self.src_points, dst_points, dst_shape, dst_img, self.args, 68)
            self.writer.write(dst_img)
            if self.args.show:
                st.image(dst_img, channels="BGR")

        self.video.release()
        self.writer.release()
        cv2.destroyAllWindows()

# 設置日誌的格式
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(lineno)d:%(message)s")

# 創建網頁的標題
st.title('Face Swap Video')

# 創建文件上傳的選項
uploaded_source_file = st.file_uploader("Source Image", type=['jpg', 'png', 'jpeg'])
uploaded_video_file = st.file_uploader("Video", type=['mp4', 'avi', 'mov'])

# 創建一個佔位符，用於顯示源圖片
source_image_placeholder = st.empty()

# 創建一個佔位符，用於顯示視頻
video_placeholder = st.empty()

# 創建一個佔位符，用於顯示輸出視頻的路徑
save_path_placeholder = st.empty()

# 創建一個按鈕，用於開始人臉交換
start_button = st.button('Start Face Swap')

# 如果上傳了源圖片和視頻，則進行以下操作
if uploaded_source_file is not None and uploaded_video_file is not None:
    # 將源圖片轉換為numpy數組
    source_pil_image = Image.open(uploaded_source_file)
    source_image = np.array(source_pil_image)
    # 獲取視頻的文件名
    video_path = uploaded_video_file.name
    # 獲取輸出視頻的默認路徑
    save_path = "output.mp4"
    # 在佔位符中顯示源圖片
    source_image_placeholder.write('Source Image:')
    source_image_placeholder.image(source_image, channels="RGB")
    # 在佔位符中顯示視頻
    video_placeholder.write('Video:')
    video_placeholder.video(uploaded_video_file)
    # 在佔位符中顯示輸出視頻的路徑
    save_path_placeholder.write('Output Video Path:')
    save_path = save_path_placeholder.text_input("", value=save_path)
    # 如果按下了按鈕，則進行以下操作
    if start_button:
        # 顯示處理中的提示
        with st.spinner('Processing...'):
            # 創建一個參數對象，用於傳遞給VideoHandler類
            args = argparse.Namespace(
                src_img=uploaded_source_file.name,
                video_path=video_path,
                warp_2d=False,
                correct_color=False,
                show=False,
                save_path=save_path
            )
            # 創建一個VideoHandler對象，並調用start方法
            VideoHandler(video_path, uploaded_source_file.name, args).start()
        # 顯示人臉交換完成的提示
        st.success('Face swap completed!')
else:
    # 如果沒有上傳源圖片和視頻，則顯示提示
    st.write('Please upload a source image and a video file to start the face swap process.')
        
