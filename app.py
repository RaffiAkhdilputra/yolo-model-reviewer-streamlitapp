import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path
import requests
from io import BytesIO
from core.model_manager import ModelManager, MODELS_DIR
from core.overlay import draw_detections, OverlayConfig

st.set_page_config(page_title="YOLO Model Reviewer", layout="wide")

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Session State for ModelManager
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager()

st.title("YOLO Model Reviewer")

# Sidebar Configuration
st.sidebar.header("Overlay Configuration")
cfg = OverlayConfig()
cfg.show_bbox = st.sidebar.checkbox("Show Bounding Box", value=True)
cfg.show_label = st.sidebar.checkbox("Show Label", value=True)
cfg.show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
cfg.show_model_name = st.sidebar.checkbox("Show Model Name", value=False)
cfg.box_thickness = st.sidebar.slider("Box Thickness", 1, 10, 2)

tab1, tab2 = st.tabs(["Try Models", "Integration Guide"])

with tab1:
    st.header("1. Upload & Manage Models")
    uploaded_models = st.file_uploader(
        "Upload YOLO models (.pt or .onnx)", 
        type=["pt", "onnx"], 
        accept_multiple_files=True
    )
    
    if uploaded_models:
        new_uploads = False
        for uploaded_file in uploaded_models:
            file_path = MODELS_DIR / uploaded_file.name
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_uploads = True
        
        if new_uploads:
            st.success("New models uploaded successfully!")
            st.session_state.model_manager.reload()

    mm = st.session_state.model_manager
    if not mm.loaded:
        st.info("No models loaded. Please upload a model above.")
    else:
        st.subheader("Loaded Models")
        for m in mm.models:
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                m.enabled = st.checkbox(f"Enable {m.name}", value=m.enabled, key=f"enable_{m.name}")
            with col2:
                st.write(f"Task: `{m.task}` | Backend: `{m._backend}`")
            with col3:
                m.conf_threshold = st.slider(f"Confidence ({m.name})", 0.0, 1.0, float(m.conf_threshold), 0.05, key=f"conf_{m.name}")

        st.header("2. Try Models (Inference)")
        image_source = st.radio("Image Source", ["Upload File", "Camera", "URL"])
        
        uploaded_image = None
        if image_source == "Upload File":
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        elif image_source == "Camera":
            uploaded_image = st.camera_input("Take a picture")
        else:
            image_url = st.text_input("Enter Image URL")
            if image_url:
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()
                    uploaded_image = BytesIO(response.content)
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")

        if uploaded_image is not None:
            # Convert uploaded image to OpenCV format
            image = Image.open(uploaded_image).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("Running inference..."):
                start_time = time.time()
                detections = mm.run_all(frame_bgr)
                inference_time = time.time() - start_time

            # Draw detections
            output_bgr = draw_detections(frame_bgr, detections, cfg)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

            st.image(output_rgb, caption=f"Processed Image ({inference_time:.3f}s)", use_container_width=True)

            st.header("3. Evaluate (Log Output)")
            from collections import defaultdict
            grouped_dets = defaultdict(list)
            for d in detections:
                grouped_dets[d.bbox].append({
                    "model": d.model_name,
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "color_bgr": d.color
                })

            nested_log_data = []
            for bbox, dets in grouped_dets.items():
                nested_log_data.append({
                    "bbox": bbox,
                    "detections": dets
                })
            
            with st.expander("View Raw Detection Logs", expanded=True):
                st.write(f"**Total Objects Detected:** {len(grouped_dets)}")
                st.write(f"**Total Individual Classifications:** {len(detections)}")
                st.write(f"**Total Inference Time:** {inference_time:.3f} seconds")
                st.json(nested_log_data)


with tab2:
    st.header("Integration Guide")
    st.markdown("""
    This guide explains how to integrate the uploaded YOLO models into your own Python applications.

    ### 1. Using Ultralytics (`.pt` models)
    If you are using the `.pt` format models, the easiest way to run inference is using the `ultralytics` package.
    
    ```python
    from ultralytics import YOLO
    import cv2

    # Load the model
    model = YOLO("models/your_model.pt")

    # Read an image
    frame = cv2.imread("test_image.jpg")

    # Run inference
    results = model(frame, conf=0.4)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            print(f"Detected {label} at {[x1, y1, x2, y2]} with confidence {conf}")
    ```

    ### 2. Using OpenCV DNN (`.onnx` models)
    If you exported your models to `.onnx` and want to run them without PyTorch, you can use OpenCV's DNN module.
    
    ```python
    import cv2
    import numpy as np

    # Load the model
    net = cv2.dnn.readNetFromONNX("models/your_model.onnx")

    # Read and prepare image
    frame = cv2.imread("test_image.jpg")
    input_size = 640
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    
    # Run inference
    net.setInput(blob)
    outputs = net.forward()

    # Note: Parsing the raw outputs depends heavily on your YOLO export format.
    # Check the ModelManager implementation in core/model_manager.py for a robust parser.
    ```
    
    ### 3. Using the `ModelManager` class
    If you want to use the exact same logic that powers this Streamlit App, simply copy the `core/` folder into your project.
    
    ```python
    import cv2
    from core.model_manager import ModelManager
    from core.overlay import draw_detections, OverlayConfig

    # Initialize manager (automatically loads models from the models/ directory)
    manager = ModelManager()

    # Read image
    frame = cv2.imread("test_image.jpg")

    # Run inference across all loaded models
    detections = manager.run_all(frame)

    # Draw the results on the frame
    cfg = OverlayConfig(show_bbox=True, show_label=True)
    output_frame = draw_detections(frame, detections, cfg)

    # Display
    cv2.imshow("Output", output_frame)
    cv2.waitKey(0)
    ```

    ### 4. Extracting Log Information (JSON)
    To consume the raw data in your own application (e.g., to build an API response or save to a database), you can easily extract the structured details from the returned `Detection` objects:
    
    ```python
    import json
    
    # After running inference as shown above:
    # detections = manager.run_all(frame)
    
    from collections import defaultdict
    grouped_dets = defaultdict(list)
    for d in detections:
        grouped_dets[d.bbox].append({
            "model": d.model_name,
            "label": d.label,
            "confidence": round(d.confidence, 4),
            "color_bgr": d.color
        })

    nested_log_data = []
    for bbox, dets in grouped_dets.items():
        nested_log_data.append({
            "bbox": bbox,  # (x1, y1, x2, y2)
            "detections": dets
        })
        
    json_output = json.dumps(nested_log_data, indent=2)
    print("Nested Detection Logs:")
    print(json_output)
    ```
    """)

st.caption('Created by Raffi Akhdilputra | [GitHub](https://github.com/raffi-akhdilputra)')