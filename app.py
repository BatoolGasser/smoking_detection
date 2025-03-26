import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import requests
import os
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="0DxaF5zFUiG19ajI7eKN")
project = rf.workspace().project("smoking-tasfx-yjw4l")
model = project.version(2).model

# Custom API endpoint
API_URL = "https://aimicromind-platform-2025.onrender.com/api/v1/prediction/ef770add-e983-42e1-91b0-16bb4db573e7"

st.title("🚭 Real-time Smoking Detection System")

def send_alert(message):
    try:
        response = requests.post(API_URL, json={"question": message})
        return response.json()
    except Exception as e:
        st.error(f"Alert API Error: {str(e)}")
        return None

def process_image(image):
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name)
        # Run prediction
        prediction = model.predict(tmp_file.name, confidence=40).json()
        # Save prediction visualization
        model.predict(tmp_file.name).save(tmp_file.name.replace(".jpg", "_prediction.jpg"))
    
    return prediction, tmp_file.name.replace(".jpg", "_prediction.jpg")

# def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = output_file.name
    
    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create temporary frame file
        fd, frame_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)  # Close descriptor immediately
        
        try:
            # Save frame to temporary file
            cv2.imwrite(frame_path, frame)
            
            # Get prediction
            prediction = model.predict(frame_path).json()
            
            # Draw predictions on frame
            for p in prediction['predictions']:
                if p['class'] == 'Cigarette' and p['confidence'] > 0.5:
                    x1 = int(p['x'] - p['width']/2)
                    y1 = int(p['y'] - p['height']/2)
                    x2 = int(p['x'] + p['width']/2)
                    y2 = int(p['y'] + p['height']/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Cigarette {p['confidence']:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            out.write(frame)
        finally:
            # Clean up temporary frame file
            if os.path.exists(frame_path):
                os.unlink(frame_path)
    
    cap.release()
    out.release()
    return output_path 


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = output_file.name
    
    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as frame_file:
            frame_path = frame_file.name
            cv2.imwrite(frame_path, frame)
            
            # Run prediction
            prediction = model.predict(frame_path).json()
        
        # Draw predictions on frame
        for p in prediction['predictions']:
            if p['confidence'] > 0.5:  # Confidence threshold
                x1 = int(p['x'] - p['width']/2)
                y1 = int(p['y'] - p['height']/2)
                x2 = int(p['x'] + p['width']/2)
                y2 = int(p['y'] + p['height']/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{p['class']} {p['confidence']:.2f}",
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path

def process_frame(frame):
    # Create temporary file with proper handling
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "frame.jpg")
        cv2.imwrite(temp_path, frame)
        
        # Run prediction using file path
        prediction = model.predict(temp_path).json()
    
    # Convert to RGB for drawing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw predictions
    for obj in prediction.get('predictions', []):
        if obj['class'] == 'Cigarette' and obj['confidence'] > 0.5:
            x = int(obj['x'])
            y = int(obj['y'])
            width = int(obj['width'])
            height = int(obj['height'])
            
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_rgb, 
                       f"Cigarette {obj['confidence']:.2f}",
                       (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
            
            send_alert("Cigarette detected in live stream!")
    
    return frame_rgb

# Sidebar controls
st.sidebar.header("Input Options")
app_mode = st.sidebar.radio("Choose Input Source",
                           ["Image Upload", "Live Camera Detection"])

if app_mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)     
        prediction, pred_path = process_image(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(pred_path, caption="Detection Result", use_column_width=True)
        
        for p in prediction['predictions']:
            if p['class'] == 'Cigarette' and p['confidence'] > 0.5:
                st.error("🚨 Cigarette Detected!")
                send_alert("Cigarette detected in uploaded image")

# elif app_mode == "Video Upload":
#     uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
#     if uploaded_video is not None:
#         # Save uploaded video to temp file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
#             tfile.write(uploaded_video.read())
#             temp_path = tfile.name
        
#         # Display original video
#         st.video(temp_path)
        
#         # Process video
#         with st.spinner("Detecting smoking in video..."):
#             output_path = process_video(temp_path)
#             st.success("Processing complete!")
            
#             # Display result
#             st.video(output_path)
            
#             # Cleanup temp files
#             os.unlink(temp_path)
#             os.unlink(output_path)

elif app_mode == "Live Camera Detection":
    run_camera = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Process frame and get annotated image
        processed_frame = process_frame(frame)
        
        # Display the processed frame
        FRAME_WINDOW.image(processed_frame)

    camera.release()