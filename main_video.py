#! /usr/bin/env python
import os
import cv2
import logging
import argparse
import streamlit as st
import numpy as np

from face_detection import select_face
from face_swap import face_swap


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


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(lineno)d:%(message)s")

st.title('Face Swap Video')

uploaded_source_file = st.file_uploader("Source Image", type=['jpg', 'png', 'jpeg'])
uploaded_video_file = st.file_uploader("Video", type=['mp4', 'avi', 'mov'])

if uploaded_source_file is not None and uploaded_video_file is not None:
    source_image = cv2.imread(uploaded_source_file.name)
    video_path = uploaded_video_file.name
    save_path = st.text_input("Output Video Path", value="output.mp4")

    st.write('Source Image:')
    st.image(source_image, channels="BGR")

    st.write('Video:')
    st.video(uploaded_video_file)

    if st.button('Start Face Swap'):
        with st.spinner('Processing...'):
            args = argparse.Namespace(
                src_img=uploaded_source_file.name,
                video_path=video_path,
                warp_2d=False,
                correct_color=False,
                show=False,
                save_path=save_path
            )
            VideoHandler(video_path, uploaded_source_file.name, args).start()
        st.success('Face swap completed!')

st.write('Please upload a source image and a video file to start the face swap process.')
