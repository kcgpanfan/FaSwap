#! /usr/bin/env python
import os
import cv2
import numpy
import argparse
import streamlit as st
from PIL import Image, ImageEnhance

from face_detection import select_face, select_all_faces
from face_swap import face_swap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--correct_color', default=True, action='store_true', help='Correct color')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    args = parser.parse_args()

    uploaded_source_file = st.file_uploader("Source File", type=['jpg', 'png', 'jpeg'])
    uploaded_target_file = st.file_uploader("Target File", type=['jpg', 'png', 'jpeg'])

    src_face = None  # 添加这一行以确保在条件之外定义'src_face'

    if uploaded_source_file is not None and uploaded_target_file is not None:
        source_image = Image.open(uploaded_source_file)
        target_image = Image.open(uploaded_target_file)

        # Convert images from PIL to CV2
        src_img = cv2.cvtColor(numpy.array(source_image), cv2.IMREAD_COLOR)
        dst_img = cv2.cvtColor(numpy.array(target_image), cv2.IMREAD_COLOR)

        # Select src face
        src_points, src_shape, src_face = select_face(src_img)
        # Select dst face
        dst_faceBoxes = select_all_faces(dst_img)

        if dst_faceBoxes is None:
            print('Detect 0 Face !!!')
            exit(-1)

        output = dst_img
        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points,
                               dst_face["points"], dst_face["shape"],
                               output, args)

            st.markdown('<p style="text-align: left;">Result</p>', unsafe_allow_html=True)
            st.image(output, width=500)

    uploaded_video_file = st.file_uploader("Video File", type=['mp4', 'avi', 'mov'])

    if uploaded_video_file is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as file:
            file.write(uploaded_video_file.read())

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output_video.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dst_faceBoxes = select_all_faces(frame)

            if dst_faceBoxes is not None:
                for k, dst_face in dst_faceBoxes.items():
                    output = face_swap(src_face, dst_face["face"], src_points,
                                       dst_face["points"], dst_face["shape"],
                                       frame, args)
                    out.write(output)
            else:
                out.write(frame)

        cap.release()
        out.release()

        st.markdown('<p style="text-align: left;">Result</p>', unsafe_allow_html=True)
        st.video("output_video.mp4")
