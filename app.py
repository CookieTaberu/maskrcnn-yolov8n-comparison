import streamlit as st
import time
import uuid
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Try to import Detectron2, fallback if not available
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    DETECTRON2_AVAILABLE = True
except ImportError as e:
    st.error(f"Detectron2 not available: {e}")
    st.info("Running in YOLO-only mode. Only YOLOv8n will be available.")
    DETECTRON2_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Object Detection Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CONF_THRESHOLD = 0.5

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8n model with caching"""
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_detectron_model():
    """Load Mask R-CNN model with caching"""
    if not DETECTRON2_AVAILABLE:
        return None
    
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return DefaultPredictor(cfg)
    except Exception as e:
        st.error(f"Error loading Detectron2 model: {e}")
        return None

def download_image(url):
    """Download image from URL and return local filename"""
    try:
        filename = f"temp_{uuid.uuid4().hex[:6]}.jpg"
        urllib.request.urlretrieve(url, filename)
        return filename
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def process_yolo_results(results):
    """Process YOLO detection results"""
    data = {}
    
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            
            if label not in data:
                data[label] = {"count": 0, "confidences": [], "avg_conf": 0}
            
            data[label]["count"] += 1
            data[label]["confidences"].append(conf)
    
    # Calculate average confidence
    for label in data:
        data[label]["avg_conf"] = np.mean(data[label]["confidences"])
    
    return data

def process_detectron_results(outputs):
    """Process Detectron2 detection results"""
    data = {}
    instances = outputs["instances"]
    
    if len(instances) > 0:
        classes = instances.pred_classes
        scores = instances.scores
        
        # Get class names
        try:
            class_names = MetadataCatalog.get("coco_2017_train").thing_classes
        except:
            class_names = [f"class_{i}" for i in range(80)]  # COCO has 80 classes
        
        for cls, score in zip(classes, scores):
            label = class_names[int(cls)]
            conf = float(score)
            
            if label not in data:
                data[label] = {"count": 0, "confidences": [], "avg_conf": 0}
            
            data[label]["count"] += 1
            data[label]["confidences"].append(conf)
    
    # Calculate average confidence
    for label in data:
        data[label]["avg_conf"] = np.mean(data[label]["confidences"])
    
    return data

def run_yolo_detection(image_path):
    """Run YOLOv8n detection"""
    model = load_yolo_model()
    
    start_time = time.time()
    results = model(image_path)[0]
    inference_time = time.time() - start_time
    
    # Process results
    data = process_yolo_results(results)
    
    # Get annotated image
    annotated_img = results.plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return data, inference_time, annotated_img_rgb

def run_detectron_detection(image_path):
    """Run Mask R-CNN detection"""
    if not DETECTRON2_AVAILABLE:
        return {}, 0, None
    
    predictor = load_detectron_model()
    if predictor is None:
        return {}, 0, None
    
    try:
        img = cv2.imread(image_path)
        start_time = time.time()
        outputs = predictor(img)
        inference_time = time.time() - start_time
        
        # Process results
        data = process_detectron_results(outputs)
        
        # Create visualization
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("coco_2017_train"), scale=1.2)
        instances = outputs["instances"]
        out = v.draw_instance_predictions(instances.to("cpu"))
        annotated_img = out.get_image()
        
        return data, inference_time, annotated_img
    except Exception as e:
        st.error(f"Error in Detectron2 detection: {e}")
        return {}, 0, None

def format_results_table(data, model_name, inference_time):
    """Format detection results as a table"""
    if not data:
        return f"**{model_name}** - No objects detected (Time: {inference_time:.3f}s)"
    
    total_objects = sum(stats['count'] for stats in data.values())
    
    result_text = f"**{model_name}** - {total_objects} objects detected (Time: {inference_time:.3f}s)\n\n"
    
    # Create table data
    table_data = []
    for label, stats in sorted(data.items(), key=lambda x: x[1]['count'], reverse=True):
        table_data.append({
            "Object": label,
            "Count": stats['count'],
            "Avg Confidence": f"{stats['avg_conf']:.3f}"
        })
    
    return result_text, table_data

def main():
    st.title("🔍 Object Detection Comparison")
    
    if DETECTRON2_AVAILABLE:
        st.markdown("**YOLOv8n vs Mask R-CNN**")
    else:
        st.markdown("**YOLOv8n Object Detection** (Detectron2 unavailable)")
        st.warning("⚠️ Detectron2 is not available. Only YOLOv8n detection will work.")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Image URL"]
    )
    
    # Confidence threshold
    global CONF_THRESHOLD
    CONF_THRESHOLD = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Model information
    with st.sidebar.expander("Model Information"):
        st.markdown("""
        **YOLOv8n:**
        - Fast inference
        - Good for real-time detection
        - Lightweight model
        
        **Mask R-CNN:**
        - High accuracy
        - Instance segmentation
        - Heavier model
        """)
    
    # Main content
    image_path = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file for object detection"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.read())
                image_path = tmp_file.name
    
    else:  # Image URL
        url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/image.jpg",
            help="Enter a direct URL to an image file"
        )
        
        if url and url.strip():
            if st.button("Download Image"):
                with st.spinner("Downloading image..."):
                    image_path = download_image(url)
    
    # Process image if available
    if image_path and os.path.exists(image_path):
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Original Image")
            original_img = Image.open(image_path)
            st.image(original_img, use_column_width=True)
        
        # Run detection button
        if st.button("🚀 Run Object Detection", type="primary"):
            with st.spinner("Running object detection..."):
                try:
                    # Run YOLO model
                    yolo_data, yolo_time, yolo_img = run_yolo_detection(image_path)
                    
                    # Run Detectron2 model if available
                    if DETECTRON2_AVAILABLE:
                        detectron_data, detectron_time, detectron_img = run_detectron_detection(image_path)
                    else:
                        detectron_data, detectron_time, detectron_img = {}, 0, None
                    
                    # Display results
                    st.header("Detection Results")
                    
                    if DETECTRON2_AVAILABLE and detectron_img is not None:
                        # Performance comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "YOLOv8n Speed", 
                                f"{yolo_time:.3f}s",
                                delta=f"{yolo_time - detectron_time:.3f}s" if yolo_time < detectron_time else None
                            )
                        
                        with col2:
                            st.metric(
                                "Mask R-CNN Speed", 
                                f"{detectron_time:.3f}s",
                                delta=f"{detectron_time - yolo_time:.3f}s" if detectron_time < yolo_time else None
                            )
                        
                        # Detection results side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("YOLOv8n Results")
                            st.image(yolo_img, use_column_width=True)
                            
                            yolo_text, yolo_table = format_results_table(yolo_data, "YOLOv8n", yolo_time)
                            st.markdown(yolo_text)
                            if yolo_table:
                                st.dataframe(yolo_table, use_container_width=True)
                        
                        with col2:
                            st.subheader("Mask R-CNN Results")
                            st.image(detectron_img, use_column_width=True)
                            
                            detectron_text, detectron_table = format_results_table(detectron_data, "Mask R-CNN", detectron_time)
                            st.markdown(detectron_text)
                            if detectron_table:
                                st.dataframe(detectron_table, use_container_width=True)
                        
                        # Summary comparison
                        st.header("Summary Comparison")
                        
                        yolo_total = sum(stats['count'] for stats in yolo_data.values())
                        detectron_total = sum(stats['count'] for stats in detectron_data.values())
                        
                        summary_data = {
                            "Model": ["YOLOv8n", "Mask R-CNN"],
                            "Inference Time (s)": [f"{yolo_time:.3f}", f"{detectron_time:.3f}"],
                            "Total Objects": [yolo_total, detectron_total],
                            "Speed Rank": ["🥇" if yolo_time < detectron_time else "🥈", 
                                         "🥇" if detectron_time < yolo_time else "🥈"],
                            "Detection Rank": ["🥇" if yolo_total > detectron_total else "🥈", 
                                             "🥇" if detectron_total > yolo_total else "🥈"]
                        }
                        
                        st.dataframe(summary_data, use_container_width=True)
                        
                        # Recommendations
                        st.header("Recommendations")
                        
                        if yolo_time < detectron_time:
                            st.success("🚀 **YOLOv8n** is faster - better for real-time applications")
                        else:
                            st.info("🎯 **Mask R-CNN** provides more detailed analysis")
                        
                        if yolo_total > detectron_total:
                            st.info("🔍 **YOLOv8n** detected more objects")
                        elif detectron_total > yolo_total:
                            st.info("🔍 **Mask R-CNN** detected more objects")
                        else:
                            st.info("🤝 Both models detected the same number of objects")
                    
                    else:
                        # YOLO-only mode
                        st.subheader("YOLOv8n Results")
                        st.image(yolo_img, use_column_width=True)
                        
                        yolo_text, yolo_table = format_results_table(yolo_data, "YOLOv8n", yolo_time)
                        st.markdown(yolo_text)
                        if yolo_table:
                            st.dataframe(yolo_table, use_container_width=True)
                        
                        st.success("🚀 **YOLOv8n** detection completed successfully!")
                
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.exception(e)
        
        # Clean up temporary file
        if input_method == "Upload Image":
            try:
                os.unlink(image_path)
            except:
                pass
    
    else:
        st.info("👆 Please upload an image or provide an image URL to start object detection comparison.")
        
        # Example images
        st.subheader("Try these example images:")
        example_urls = [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800",
            "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=800"
        ]
        
        for i, url in enumerate(example_urls):
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.experimental_set_query_params(url=url)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
