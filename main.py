import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils import Tracker

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Real-Time Object Monitoring", layout="wide")

model = YOLO("yolov8n.pt")

tracker = Tracker()

st.title("Real-Time Missing and New Object Detection 🚀")
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.01)
frame_skip = st.sidebar.slider("Frame Skip (for performance)", 1, 5, 2)

run = st.sidebar.checkbox('Start Detection')

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        bboxes = []
        for det in detections:
            if len(det) == 6:
                x1, y1, x2, y2, conf, cls = det
            else:
                x1, y1, x2, y2 = det
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])


        frame, events = tracker.update(frame, bboxes)

        for event in events:
            cv2.putText(frame, event, (10, 30 + 30 * events.index(event)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()