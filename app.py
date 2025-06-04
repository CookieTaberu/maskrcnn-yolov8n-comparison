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

# Set page config
st.set_page_config(
    page_title="YOLOv8n vs Mask R-CNN Comparison",
    page_icon="🔍",
    layout="wide"
)

# Generic class naming without COCO labels

@st.cache_resource
def load_models():
    """Load YOLOv8n and Mask R-CNN models"""
    try:
        # Load YOLOv8n
        yolo_model = YOLO('yolov8n.pt')
        
        # Load Mask R-CNN
        maskrcnn_model = detection.maskrcnn_resnet50_fpn(pretrained=True)
        maskrcnn_model.eval()
        
        return yolo_model, maskrcnn_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_image_maskrcnn(image):
    """Preprocess image for Mask R-CNN"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def run_yolo_detection(model, image, confidence_threshold=0.5):
    """Run YOLOv8n detection"""
    start_time = time.time()
    
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(opencv_image, conf=confidence_threshold)
    
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
                
                # Get class name (generic)
                class_name = f"class_{class_id}"
                
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

def run_maskrcnn_detection(model, image, confidence_threshold=0.5):
    """Run Mask R-CNN detection"""
    start_time = time.time()
    
    # Preprocess image
    input_tensor = preprocess_image_maskrcnn(image)
    
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
        class_name = f"class_{label}"
        
        detections.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(score),
            'class': class_name,
            'class_id': int(label)
        })
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(annotated_image, f"{class_name}: {score:.2f}", 
                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return detections, annotated_image, inference_time

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
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Overall statistics
    stats_data = {
        'Metric': [
            'Total Detections',
            'Inference Time (s)',
            'Average Confidence (YOLOv8n)',
            'Average Confidence (Mask R-CNN)',
            'Unique Classes Detected'
        ],
        'YOLOv8n': [
            len(yolo_detections),
            f"{yolo_time:.4f}",
            f"{np.mean([d['confidence'] for d in yolo_detections]):.3f}" if yolo_detections else "N/A",
            "N/A",
            len(yolo_classes)
        ],
        'Mask R-CNN': [
            len(maskrcnn_detections),
            f"{maskrcnn_time:.4f}",
            "N/A",
            f"{np.mean([d['confidence'] for d in maskrcnn_detections]):.3f}" if maskrcnn_detections else "N/A",
            len(maskrcnn_classes)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    return comparison_df, stats_df

def main():
    st.title("🔍 YOLOv8n vs Mask R-CNN Object Detection Comparison")
    st.markdown("Upload an image to compare object detection performance between YOLOv8n and Mask R-CNN models.")
    
    # Load models
    with st.spinner("Loading models..."):
        yolo_model, maskrcnn_model = load_models()
    
    if yolo_model is None or maskrcnn_model is None:
        st.error("Failed to load models. Please check your installation.")
        st.markdown("""
        **Installation Requirements:**
        ```bash
        pip install streamlit ultralytics torch torchvision opencv-python pillow matplotlib pandas numpy
        ```
        """)
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar for parameters
    st.sidebar.header("Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image for object detection comparison"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        st.subheader("📸 Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Run detections
        with st.spinner("Running object detection..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 YOLOv8n Results")
                yolo_detections, yolo_annotated, yolo_time = run_yolo_detection(
                    yolo_model, image, confidence_threshold
                )
                st.image(yolo_annotated, caption=f"YOLOv8n Detection (Time: {yolo_time:.4f}s)", use_column_width=True)
                st.write(f"**Detections found:** {len(yolo_detections)}")
            
            with col2:
                st.subheader("🔴 Mask R-CNN Results")
                maskrcnn_detections, maskrcnn_annotated, maskrcnn_time = run_maskrcnn_detection(
                    maskrcnn_model, image, confidence_threshold
                )
                st.image(maskrcnn_annotated, caption=f"Mask R-CNN Detection (Time: {maskrcnn_time:.4f}s)", use_column_width=True)
                st.write(f"**Detections found:** {len(maskrcnn_detections)}")
        
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
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
            
            ax2.bar(models, times, color=colors, alpha=0.7)
            ax2.set_ylabel('Inference Time (seconds)')
            ax2.set_title('Inference Time Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(times):
                ax2.text(i, v + max(times) * 0.01, f'{v:.4f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Detailed detection results
        with st.expander("🔍 Detailed Detection Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### YOLOv8n Detections")
                if yolo_detections:
                    yolo_df = pd.DataFrame(yolo_detections)
                    st.dataframe(yolo_df)
                else:
                    st.write("No objects detected")
            
            with col2:
                st.markdown("#### Mask R-CNN Detections")
                if maskrcnn_detections:
                    maskrcnn_df = pd.DataFrame(maskrcnn_detections)
                    st.dataframe(maskrcnn_df)
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
        
        **🔴 Mask R-CNN (Region-based CNN)**
        - More accurate but slower detection
        - Two-stage detection architecture  
        - Better for precision-critical applications
        
        **Features:**
        - Side-by-side visual comparison
        - Performance metrics (speed, accuracy)
        - Class-wise detection analysis
        - Detailed detection results
        """)

if __name__ == "__main__":
    main()
