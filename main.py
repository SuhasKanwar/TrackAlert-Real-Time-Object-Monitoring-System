import streamlit as st
import cv2
from ultralytics import YOLO
from utils import Tracker
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Real-Time Object Monitoring", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

tracker = Tracker()

st.title("Real-Time Missing and New Object Detection 🚀")

st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.01)
frame_skip = st.sidebar.slider("Frame Skip (for performance)", 1, 5, 2)
max_disappeared = st.sidebar.slider("Max Disappeared Frames", 10, 100, 30)
tracker.max_disappeared = max_disappeared

source_option = st.sidebar.selectbox("Select Input Source", ["Webcam", "Video File"])

video_file = None
if source_option == "Video File":
    video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

show_bboxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_centroids = st.sidebar.checkbox("Show Centroids", True)
show_ids = st.sidebar.checkbox("Show Object IDs", True)

run = st.sidebar.checkbox('Start Detection')

col1, col2 = st.columns([3, 1])

with col1:
    FRAME_WINDOW = st.image([])
    fps_text = st.empty()

with col2:
    st.subheader("Object Status")
    current_objects = st.empty()
    missing_objects = st.empty()
    st.subheader("Events")
    events_display = st.empty()

class_names = model.names
active_objects = {}
departed_objects = {}

if run:
    if source_option == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_file is not None:
            temp_file = f"temp_video_{time.time()}.mp4"
            with open(temp_file, "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture(temp_file)
        else:
            st.error("Please upload a video file.")
            run = False
            cap = None
    
    if cap is not None and cap.isOpened():
        frame_count = 0
        fps_list = []
        start_time = time.time()
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("End of video or failed to read from source.")
                break

            current_time = time.time()
            fps = 1 / (current_time - start_time) if (current_time - start_time) > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            start_time = current_time

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))

            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()

            bboxes = []
            class_ids = []
            for det in detections:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    class_ids.append(int(cls))
                else:
                    x1, y1, x2, y2 = det[:4]
                    class_ids.append(-1)
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])

            frame, events, current_ids, deregistered_ids = tracker.update(frame, bboxes, class_ids, class_names, show_bboxes, show_centroids, show_ids)

            for obj_id in current_ids:
                if obj_id in active_objects:
                    active_objects[obj_id]['last_seen'] = time.time()
                else:
                    active_objects[obj_id] = {
                        'class': current_ids[obj_id]['class'],
                        'time': time.time(),
                        'last_seen': time.time()
                    }
            
            for obj_id in deregistered_ids:
                if obj_id in active_objects:
                    departed_objects[obj_id] = {
                        'class': active_objects[obj_id]['class'],
                        'time_appeared': active_objects[obj_id]['time'],
                        'time_left': time.time(),
                        'duration': time.time() - active_objects[obj_id]['time']
                    }
                    del active_objects[obj_id]

            fps_text.text(f"FPS: {avg_fps:.2f}")
            
            active_text = "| ID | Class | Time Visible |\n|---|-------|-------------|\n"
            for obj_id, data in active_objects.items():
                duration = time.time() - data['time']
                active_text += f"| {obj_id} | {data['class']} | {duration:.1f}s |\n"
            current_objects.markdown(active_text)
            
            departed_text = "| ID | Class | Duration Visible |\n|---|-------|----------------|\n"
            for obj_id, data in list(departed_objects.items())[-5:]:  # Show last 5 only
                departed_text += f"| {obj_id} | {data['class']} | {data['duration']:.1f}s |\n"
            missing_objects.markdown(departed_text)
            
            events_text = ""
            for event in events[-10:]:
                events_text += f"• {event}\n"
            events_display.text(events_text)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        
        cap.release()
        if source_option == "Video File" and video_file is not None:
            try:
                os.remove(temp_file)
            except:
                pass