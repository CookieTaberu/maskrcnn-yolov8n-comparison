import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import os
import warnings
import tempfile
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Set YOLO config directory to avoid permission issues
os.environ.setdefault('YOLO_CONFIG_DIR', tempfile.gettempdir())

# Set page config
st.set_page_config(
    page_title="YOLOv8n vs Mask R-CNN Comparison",
    page_icon="🔍",
    layout="wide"
)

# COCO class names for better labeling
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource
def load_models():
    """Load YOLOv8n and Mask R-CNN models with better error handling"""
    try:
        # Suppress YOLO verbose output
        import logging
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        # Load YOLOv8n
        with st.spinner("Loading YOLOv8n model..."):
            yolo_model = YOLO('yolov8n.pt')
        
        # Load Mask R-CNN with proper error handling
        with st.spinner("Loading Mask R-CNN model..."):
            try:
                maskrcnn_model = detection.maskrcnn_resnet50_fpn(weights='COCO_V1')
            except:
                # Fallback to older syntax
                maskrcnn_model = detection.maskrcnn_resnet50_fpn(pretrained=True)
            
            maskrcnn_model.eval()
            
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            maskrcnn_model = maskrcnn_model.to(device)
        
        return yolo_model, maskrcnn_model, device
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def get_class_name(class_id):
    """Get class name from COCO class ID"""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"

def preprocess_image_maskrcnn(image, device):
    """Preprocess image for Mask R-CNN"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)

def run_yolo_detection(model, image, confidence_threshold=0.5):
    """Run YOLOv8n detection with improved error handling"""
    try:
        start_time = time.time()
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(opencv_image, conf=confidence_threshold, verbose=False)
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        annotated_image = opencv_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = get_class_name(class_id)
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"{class_name}: {confidence:.2f}", 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return detections, annotated_image, inference_time
    
    except Exception as e:
        st.error(f"Error in YOLO detection: {str(e)}")
        return [], np.array(image), 0.0

def run_maskrcnn_detection(model, image, device, confidence_threshold=0.5):
    """Run Mask R-CNN detection with improved error handling"""
    try:
        start_time = time.time()
        
        # Preprocess image
        input_tensor = preprocess_image_maskrcnn(image, device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(input_tensor)
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        annotated_image = np.array(image)
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        valid_indices = scores > confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            class_name = get_class_name(label - 1)  # COCO labels are 1-indexed
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(score),
                'class': class_name,
                'class_id': int(label - 1)
            })
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(annotated_image, f"{class_name}: {score:.2f}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return detections, annotated_image, inference_time
    
    except Exception as e:
        st.error(f"Error in Mask R-CNN detection: {str(e)}")
        return [], np.array(image), 0.0

def create_comparison_table(yolo_detections, maskrcnn_detections, yolo_time, maskrcnn_time):
    """Create comparison table between the two models"""
    
    # Count detections by class for both models
    yolo_classes = {}
    maskrcnn_classes = {}
    
    for det in yolo_detections:
        class_name = det['class']
        yolo_classes[class_name] = yolo_classes.get(class_name, 0) + 1
    
    for det in maskrcnn_detections:
        class_name = det['class']
        maskrcnn_classes[class_name] = maskrcnn_classes.get(class_name, 0) + 1
    
    # Get all unique classes
    all_classes = set(list(yolo_classes.keys()) + list(maskrcnn_classes.keys()))
    
    # Create comparison data
    comparison_data = []
    for class_name in sorted(all_classes):
        yolo_count = yolo_classes.get(class_name, 0)
        maskrcnn_count = maskrcnn_classes.get(class_name, 0)
        comparison_data.append({
            'Class': class_name,
            'YOLOv8n Count': yolo_count,
            'Mask R-CNN Count': maskrcnn_count,
            'Difference': yolo_count - maskrcnn_count
        })
    
    comparison_df = pd.DataFrame(comparison_data) if comparison_data else pd.DataFrame()
    
    # Overall statistics
    yolo_avg_conf = np.mean([d['confidence'] for d in yolo_detections]) if yolo_detections else 0
    maskrcnn_avg_conf = np.mean([d['confidence'] for d in maskrcnn_detections]) if maskrcnn_detections else 0
    
    stats_data = {
        'Metric': [
            'Total Detections',
            'Inference Time (s)',
            'Average Confidence',
            'Unique Classes Detected',
            'Speed (FPS)'
        ],
        'YOLOv8n': [
            len(yolo_detections),
            f"{yolo_time:.4f}",
            f"{yolo_avg_conf:.3f}" if yolo_detections else "N/A",
            len(yolo_classes),
            f"{1/yolo_time:.1f}" if yolo_time > 0 else "N/A"
        ],
        'Mask R-CNN': [
            len(maskrcnn_detections),
            f"{maskrcnn_time:.4f}",
            f"{maskrcnn_avg_conf:.3f}" if maskrcnn_detections else "N/A",
            len(maskrcnn_classes),
            f"{1/maskrcnn_time:.1f}" if maskrcnn_time > 0 else "N/A"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    return comparison_df, stats_df

def create_visualizations(comparison_df, yolo_time, maskrcnn_time):
    """Create comparison visualizations"""
    if comparison_df.empty:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Detection count comparison
    classes = comparison_df['Class']
    yolo_counts = comparison_df['YOLOv8n Count']
    maskrcnn_counts = comparison_df['Mask R-CNN Count']
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax1.bar(x - width/2, yolo_counts, width, label='YOLOv8n', color='green', alpha=0.7)
    ax1.bar(x + width/2, maskrcnn_counts, width, label='Mask R-CNN', color='red', alpha=0.7)
    ax1.set_xlabel('Object Classes')
    ax1.set_ylabel('Detection Count')
    ax1.set_title('Detection Count Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Inference time comparison
    models = ['YOLOv8n', 'Mask R-CNN']
    times = [yolo_time, maskrcnn_time]
    colors = ['green', 'red']
    
    bars = ax2.bar(models, times, color=colors, alpha=0.7)
    ax2.set_ylabel('Inference Time (seconds)')
    ax2.set_title('Inference Time Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times) * 0.01, 
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Speed comparison (FPS)
    fps_values = [1/t if t > 0 else 0 for t in times]
    bars = ax3.bar(models, fps_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Speed (FPS)')
    ax3.set_title('Speed Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, fps in zip(bars, fps_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values) * 0.01, 
                f'{fps:.1f}', ha='center', va='bottom')
    
    # Detection difference
    differences = comparison_df['Difference']
    colors_diff = ['green' if d >= 0 else 'red' for d in differences]
    
    ax4.bar(classes, differences, color=colors_diff, alpha=0.7)
    ax4.set_xlabel('Object Classes')
    ax4.set_ylabel('Detection Difference (YOLO - Mask R-CNN)')
    ax4.set_title('Detection Count Difference')
    ax4.set_xticklabels(classes, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🔍 YOLOv8n vs Mask R-CNN Object Detection Comparison")
    st.markdown("Upload an image to compare object detection performance between YOLOv8n and Mask R-CNN models.")
    
    # Load models
    with st.spinner("Loading models... This may take a moment on first run."):
        yolo_model, maskrcnn_model, device = load_models()
    
    if yolo_model is None or maskrcnn_model is None:
        st.error("Failed to load models. Please check your installation.")
        st.markdown("""
        **Installation Requirements:**
        ```bash
        pip install streamlit ultralytics torch torchvision opencv-python pillow matplotlib pandas numpy
        ```
        
        **Common Issues:**
        - Make sure you have sufficient disk space for model downloads
        - Check internet connection for initial model download
        - For GPU support, install appropriate CUDA version of PyTorch
        """)
        return
    
    st.success(f"✅ Models loaded successfully! Using device: {device}")
    
    # Sidebar for parameters
    st.sidebar.header("Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    show_detailed_results = st.sidebar.checkbox(
        "Show Detailed Results", 
        value=False,
        help="Display detailed detection data tables"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image for object detection comparison"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return
        
        st.subheader("📸 Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Display image info
        st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        # Run detections
        with st.spinner("Running object detection..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 YOLOv8n Results")
                yolo_detections, yolo_annotated, yolo_time = run_yolo_detection(
                    yolo_model, image, confidence_threshold
                )
                st.image(yolo_annotated, caption=f"YOLOv8n Detection", use_container_width=True)
                st.metric("Detections", len(yolo_detections))
                st.metric("Inference Time", f"{yolo_time:.4f}s")
                st.metric("Speed", f"{1/yolo_time:.1f} FPS" if yolo_time > 0 else "N/A")
            
            with col2:
                st.subheader("🔴 Mask R-CNN Results")
                maskrcnn_detections, maskrcnn_annotated, maskrcnn_time = run_maskrcnn_detection(
                    maskrcnn_model, image, device, confidence_threshold
                )
                st.image(maskrcnn_annotated, caption=f"Mask R-CNN Detection", use_container_width=True)
                st.metric("Detections", len(maskrcnn_detections))
                st.metric("Inference Time", f"{maskrcnn_time:.4f}s")
                st.metric("Speed", f"{1/maskrcnn_time:.1f} FPS" if maskrcnn_time > 0 else "N/A")
        
        # Create and display comparison tables
        st.subheader("📊 Performance Comparison")
        
        comparison_df, stats_df = create_comparison_table(
            yolo_detections, maskrcnn_detections, yolo_time, maskrcnn_time
        )
        
        # Display overall statistics
        st.markdown("### Overall Statistics")
        st.dataframe(stats_df, use_container_width=True)
        
        # Display class-wise comparison
        if not comparison_df.empty:
            st.markdown("### Class-wise Detection Count")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            fig = create_visualizations(comparison_df, yolo_time, maskrcnn_time)
            if fig:
                st.pyplot(fig)
        else:
            st.info("No objects detected by either model at the current confidence threshold.")
        
        # Performance insights
        st.subheader("🎯 Performance Insights")
        
        if yolo_time > 0 and maskrcnn_time > 0:
            speed_ratio = maskrcnn_time / yolo_time
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Speed Advantage", 
                    f"{speed_ratio:.1f}x",
                    help="How many times faster YOLOv8n is compared to Mask R-CNN"
                )
            
            with col2:
                detection_diff = len(yolo_detections) - len(maskrcnn_detections)
                st.metric(
                    "Detection Difference", 
                    f"{detection_diff:+d}",
                    help="Difference in number of detections (YOLO - Mask R-CNN)"
                )
            
            with col3:
                if yolo_detections and maskrcnn_detections:
                    yolo_avg_conf = np.mean([d['confidence'] for d in yolo_detections])
                    maskrcnn_avg_conf = np.mean([d['confidence'] for d in maskrcnn_detections])
                    conf_diff = yolo_avg_conf - maskrcnn_avg_conf
                    st.metric(
                        "Confidence Difference", 
                        f"{conf_diff:+.3f}",
                        help="Difference in average confidence (YOLO - Mask R-CNN)"
                    )
        
        # Detailed detection results
        if show_detailed_results:
            with st.expander("🔍 Detailed Detection Results", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### YOLOv8n Detections")
                    if yolo_detections:
                        yolo_df = pd.DataFrame(yolo_detections)
                        st.dataframe(yolo_df, use_container_width=True)
                    else:
                        st.write("No objects detected")
                
                with col2:
                    st.markdown("#### Mask R-CNN Detections")
                    if maskrcnn_detections:
                        maskrcnn_df = pd.DataFrame(maskrcnn_detections)
                        st.dataframe(maskrcnn_df, use_container_width=True)
                    else:
                        st.write("No objects detected")
    
    else:
        st.info("👆 Please upload an image to start the comparison")
        
        # Show sample information
        st.markdown("""
        ### About This App
        
        This application compares two popular object detection models:
        
        **🟢 YOLOv8n (You Only Look Once v8 Nano)**
        - Fast, lightweight model optimized for speed
        - Single-shot detection architecture
        - Good for real-time applications
        - Typically 5-10x faster than Mask R-CNN
        
        **🔴 Mask R-CNN (Region-based CNN)**
        - More accurate but slower detection
        - Two-stage detection architecture  
        - Better for precision-critical applications
        - Can also perform instance segmentation
        
        **Features:**
        - Side-by-side visual comparison
        - Performance metrics (speed, accuracy, FPS)  
        - Class-wise detection analysis
        - Detailed detection results
        - Interactive confidence threshold adjustment
        
        **Supported Formats:** PNG, JPG, JPEG
        """)

if __name__ == "__main__":
    main()
