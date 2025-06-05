import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time
import base64
import numpy as np
from ultralytics import YOLO
import cv2

# === Inisialisasi session state ===
if 'history' not in st.session_state:
    st.session_state['history'] = []

# === COCO Class Names ===
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# === Model Loader ===
@st.cache_resource
def load_yolov8():
    model = YOLO('yolov8n.pt')  # Will download if not present
    return model

@st.cache_resource
def load_maskrcnn():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# === Deteksi Berdasarkan Model ===
def detect_objects_yolo(image: Image.Image, confidence_threshold: float = 0.5):
    model = load_yolov8()
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array, conf=confidence_threshold)
    
    # Extract results
    boxes = []
    labels = []
    scores = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])
                
                # Get class label
                class_id = int(box.cls[0].cpu().numpy())
                labels.append(COCO_CLASSES[class_id])
                
                # Get confidence score
                scores.append(float(box.conf[0].cpu().numpy()))
    
    return {"boxes": boxes, "labels": labels, "scores": scores}

def detect_objects_maskrcnn(image: Image.Image, confidence_threshold: float = 0.5):
    model, device = load_maskrcnn()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Extract results
    boxes = []
    labels = []
    scores = []
    
    pred = predictions[0]
    
    # Filter by confidence threshold
    valid_indices = pred['scores'] > confidence_threshold
    
    if torch.any(valid_indices):
        filtered_boxes = pred['boxes'][valid_indices].cpu().numpy()
        filtered_labels = pred['labels'][valid_indices].cpu().numpy()
        filtered_scores = pred['scores'][valid_indices].cpu().numpy()
        
        for box, label_id, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            boxes.append(box.tolist())
            labels.append(COCO_CLASSES[label_id - 1])  # COCO labels are 1-indexed
            scores.append(float(score))
    
    return {"boxes": boxes, "labels": labels, "scores": scores}

def detect_objects(image: Image.Image, model_option: str, confidence_threshold: float = 0.5):
    if model_option == "YOLOv8n":
        return detect_objects_yolo(image, confidence_threshold)
    else:
        return detect_objects_maskrcnn(image, confidence_threshold)

# === Visualisasi dan Buffering ===
def draw_boxes(image: Image.Image, boxes, labels, scores):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{label} ({score:.2f})", fontsize=10, color='white', 
                bbox=dict(facecolor=color, alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

def image_to_base64(buf):
    return base64.b64encode(buf.getvalue()).decode()

# === UI Streamlit ===
st.title("🔍 Object Detection Comparison: YOLOv8n vs Mask R-CNN")
st.markdown("Compare the performance of YOLOv8n and Mask R-CNN on your images")

# Model selection
col1, col2 = st.columns(2)
with col1:
    model_option = st.selectbox("Pilih Model", ["YOLOv8n", "Mask R-CNN"])

with col2:
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.info(f"**{model_option}** - Confidence threshold: {confidence_threshold}")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

st.divider()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    st.divider()

    if st.button("🚀 Deteksi Objek", type="primary"):
        with st.spinner(f"Mendeteksi objek menggunakan {model_option}..."):
            start_time = time.time()
            result = detect_objects(image, model_option, confidence_threshold)
            end_time = time.time()
            duration = end_time - start_time

            boxes = result['boxes']
            labels = result['labels']
            scores = result['scores']

            st.success(f"✅ Deteksi selesai dalam {duration:.2f} detik!")

            if boxes:
                img_buf = draw_boxes(image, boxes, labels, scores)
                st.image(img_buf, caption=f"Hasil Deteksi - {model_option}", use_container_width=True)

                st.header("📊 Objek Terdeteksi")
                
                # Create a nice table for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Daftar Objek:")
                    for i, (label, score) in enumerate(zip(labels, scores), 1):
                        st.markdown(f"**{i}.** {label} - *{score:.3f}*")
                
                with col2:
                    st.subheader("Statistik:")
                    st.metric("Total Objek", len(labels))
                    st.metric("Rata-rata Confidence", f"{np.mean(scores):.3f}")
                    st.metric("Confidence Tertinggi", f"{max(scores):.3f}")

                # Model info
                st.info(f"🤖 **Model**: {model_option} | ⏱️ **Waktu**: {duration:.2f}s | 🎯 **Threshold**: {confidence_threshold}")

                # Simpan ke history (maksimal 10 item)
                st.session_state['history'].insert(0, {
                    "model": model_option,
                    "confidence": confidence_threshold,
                    "duration": duration,
                    "labels": list(zip(labels, scores)),
                    "image_b64": image_to_base64(img_buf),
                    "total_objects": len(labels)
                })
                st.session_state['history'] = st.session_state['history'][:10]
            else:
                st.warning(f"Tidak ada objek yang terdeteksi dengan confidence threshold {confidence_threshold}")
                st.info("Coba turunkan confidence threshold atau gunakan gambar yang berbeda")

# === Sidebar History ===
with st.sidebar:
    st.header("📈 Riwayat Prediksi")
    
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state['history'] = []
        st.rerun()
    
    if st.session_state['history']:
        for i, entry in enumerate(st.session_state['history']):
            with st.expander(f"#{i+1} - {entry['model']} ({entry['total_objects']} objek)"):
                st.markdown(f"**Model**: {entry['model']}")
                st.markdown(f"**Confidence**: {entry['confidence']}")
                st.markdown(f"**Durasi**: {entry['duration']:.2f} detik")
                st.markdown(f"**Total Objek**: {entry['total_objects']}")
                
                st.markdown("**Objek Terdeteksi:**")
                for lbl, sc in entry['labels']:
                    st.markdown(f"• {lbl} ({sc:.3f})")
                
                st.image(f"data:image/png;base64,{entry['image_b64']}", use_container_width=True)
    else:
        st.info("Belum ada prediksi yang dilakukan.")

# === Footer ===
st.divider()
st.markdown("### 📋 Informasi Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **YOLOv8n:**
    - ⚡ Sangat cepat
    - 🎯 Akurasi baik
    - 💾 Model ringan (~6MB)
    - 🔧 Mudah digunakan
    """)

with col2:
    st.markdown("""
    **Mask R-CNN:**
    - 🎯 Akurasi tinggi
    - 🖼️ Segmentasi instance
    - 💾 Model besar (~170MB)
    - ⏱️ Lebih lambat
    """)

st.markdown("*Kedua model menggunakan dataset COCO dengan 80 kelas objek*")
