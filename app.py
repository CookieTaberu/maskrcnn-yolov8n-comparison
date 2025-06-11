import streamlit as st # Import streamlit first for set_page_config

# Streamlit UI Configuration - MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="Crowd Detection Comparison")

import time
import uuid
import urllib.request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from transformers import pipeline
from ultralytics import YOLO # Import YOLO from ultralytics
# PIL (Pillow) for image processing
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError # Import UnidentifiedImageError

# Configuration Constants (Defined at the top for global access)
CONF_THRESHOLD = 0.5 # Confidence threshold for displaying detections
PERSON_LABEL = "person" # The label for people in COCO dataset (class ID 0 in YOLOv8 trained on COCO)

# Initialize Models (Cached to avoid re-loading on every rerun)
@st.cache_resource
def load_yolov8_model():
    """Loads the YOLOv8n model from Ultralytics."""
    # Using 'yolov8n.pt' (nano version) as a lightweight general model
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_detr_pipeline():
    """Loads the DETR Hugging Face pipeline."""
    return pipeline("object-detection", model="facebook/detr-resnet-50")

yolov8_model = load_yolov8_model()
detr_pipeline = load_detr_pipeline()


def download_image(url, save_path=None):
    """Download image from URL"""
    if save_path is None:
        save_path = f"downloaded_{uuid.uuid4().hex[:6]}.jpg"
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        st.error(f"Failed to download image from URL: {e}. Please check the URL.")
        return None
    return save_path

def format_crowd_results(title, data, time_taken):
    """Format crowd detection results for display, focusing on 'person'."""
    summary = f"**{title}**\n\n"
    summary += f"Processing Time: `{time_taken:.2f}s`\n"
    if PERSON_LABEL not in data or not data[PERSON_LABEL]["count"]:
        summary += "> No 'person' objects detected.\n"
    else:
        person_stats = data[PERSON_LABEL]
        avg_conf = sum(person_stats["confidences"]) / len(person_stats["confidences"])
        summary += f"- **Total People Detected**: `{person_stats['count']}` (Avg. Confidence: `{avg_conf:.2f}`)\n"
        summary += f"  - `✓ Valid`: `{person_stats['true']}`, `✗ Invalid`: `{person_stats['false']}`\n"
    return summary

def run_yolov8_crowd_detection(yolo_model, image_path):
    """
    Runs YOLOv8 model on an image for crowd detection.
    Returns filtered detection data, time taken, and an annotated image focusing on 'person'.
    """
    # Image.open can directly open file paths.
    img_pil = Image.open(image_path).convert("RGB")
    
    start = time.time()
    # YOLOv8 predict method can take PIL Image directly.
    # classes=[0] filters for 'person' class (COCO dataset class ID 0).
    # conf=CONF_THRESHOLD applies the confidence threshold directly during prediction.
    results = yolo_model.predict(source=img_pil, conf=CONF_THRESHOLD, classes=[0], verbose=False) # verbose=False to suppress output
    end = time.time()

    data = {PERSON_LABEL: {"count": 0, "confidences": [], "true": 0, "false": 0}}
    annotated_img_pil = img_pil.copy()
    draw = ImageDraw.Draw(annotated_img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    person_detections = []

    # Process results. Each result object contains boxes, masks, etc.
    if results and len(results) > 0:
        for box in results[0].boxes: # Iterate through bounding boxes
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist()) # Get box coordinates
            score = box.conf.item() # Confidence score
            
            # Since we filtered by classes=[0], we know it's a person.
            # Using the actual label name for consistency in data.
            label = PERSON_LABEL 

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)

            # Put label and confidence
            text = f"{label}: {score:.2f}"
            text_y = y_min - 15 if y_min - 15 > 0 else y_min + 5
            draw.text((x_min, text_y), text, fill=(0, 255, 0), font=font)

            conf = float(score)
            valid = conf > CONF_THRESHOLD # Already filtered by predict, but keep for consistency

            data[PERSON_LABEL]["count"] += 1
            data[PERSON_LABEL]["confidences"].append(conf)
            data[PERSON_LABEL]["true" if valid else "false"] += 1

            person_detections.append({
                "label": label,
                "score": score,
                "box": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max},
                "valid": valid
            })

    return data, end - start, np.array(annotated_img_pil), person_detections

def run_detr_hf_crowd_detection(hf_pipeline, image_path):
    """
    Runs DETR (Hugging Face) model on an image for crowd detection.
    Returns filtered detection data, time taken, and an annotated image focusing on 'person'.
    This is adapted from the previous `run_hf_crowd_detection`.
    """
    img_pil = Image.open(image_path).convert("RGB")

    start = time.time()
    predictions = hf_pipeline(img_pil) # Pass PIL Image object to the pipeline
    end = time.time()

    data = {PERSON_LABEL: {"count": 0, "confidences": [], "true": 0, "false": 0}}
    annotated_img = img_pil.copy()
    draw = ImageDraw.Draw(annotated_img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    person_detections = []

    for pred in predictions:
        label = pred['label']
        score = pred['score']

        if label == PERSON_LABEL and score >= CONF_THRESHOLD: # Filter for 'person' and confidence
            box = pred['box']
            x_min, y_min, x_max, y_max = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

            # Draw bounding box using PIL
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)

            # Put label and confidence
            text = f"{label}: {score:.2f}"
            text_y = y_min - 15 if y_min - 15 > 0 else y_min + 5
            draw.text((x_min, text_y), text, fill=(0, 255, 0), font=font)

            conf = float(score)
            valid = True # Valid if it passed the threshold check above

            data[PERSON_LABEL]["count"] += 1
            data[PERSON_LABEL]["confidences"].append(conf)
            data[PERSON_LABEL]["true"] += 1
            # For DETR, if it passes the threshold, it's 'true'. No 'false' here based on this filtering.

            person_detections.append({
                "label": label,
                "score": score,
                "box": box,
                "valid": valid
            })
        elif label == PERSON_LABEL and score < CONF_THRESHOLD:
             # Count as 'false' if it's a person but below threshold
            conf = float(score)
            data[PERSON_LABEL]["count"] += 1
            data[PERSON_LABEL]["confidences"].append(conf)
            data[PERSON_LABEL]["false"] += 1


    return data, end - start, np.array(annotated_img), person_detections


def create_comparison_table(yolos_data, detr_data, yolos_time, detr_time):
    """Creates a consolidated DataFrame for model comparison."""
    # This function is correct.
    yolos_person_count = yolos_data.get(PERSON_LABEL, {}).get("count", 0)
    detr_person_count = detr_data.get(PERSON_LABEL, {}).get("count", 0)

    yolos_avg_conf = yolos_data.get(PERSON_LABEL, {}).get("confidences", [])
    yolos_avg_conf = sum(yolos_avg_conf) / len(yolos_avg_conf) if yolos_avg_conf else 0

    detr_avg_conf = detr_data.get(PERSON_LABEL, {}).get("confidences", [])
    detr_avg_conf = sum(detr_avg_conf) / len(detr_avg_conf) if detr_avg_conf else 0

    comparison_df = pd.DataFrame({
        "Metric": ["Total People Detected", "Average Confidence", "Processing Time (s)"],
        "YOLOv8 (Ultralytics)": [yolos_person_count, f"{yolos_avg_conf:.2f}", f"{yolos_time:.2f}"],
        "DETR (Hugging Face)": [detr_person_count, f"{detr_avg_conf:.2f}", f"{detr_time:.2f}"]
    })
    return comparison_df


def analyze_image_crowd(image_path_or_url=None):
    """
    Analyze image for crowd detection with YOLOv8 (Ultralytics) and DETR (Hugging Face).
    This function now expects a path to a temporary file, or a URL.
    Returns default values if any step fails.
    """
    img_path = None
    if image_path_or_url and isinstance(image_path_or_url, str) and image_path_or_url.startswith("http"):
        img_path = download_image(image_path_or_url)
    elif image_path_or_url and isinstance(image_path_or_url, str):
        img_path = image_path_or_url # It's already a local file path
    else:
        st.info("Please upload an image or provide an image URL to begin crowd detection.")
        return None, None, None, None, None, pd.DataFrame(), pd.DataFrame(), 0.0, 0.0 # Return defaults if no input

    # Initialize all return values with defaults in case of any failure below
    original_img_rgb = None
    yolov8_person_data, yolov8_time, yolov8_img_np, yolov8_raw_detections = {PERSON_LABEL: {"count": 0, "confidences": [], "true": 0, "false": 0}}, 0.0, None, []
    detr_person_data, detr_time, detr_img_np, detr_raw_detections = {PERSON_LABEL: {"count": 0, "confidences": [], "true": 0, "false": 0}}, 0.0, None, []
    yolov8_df_raw = pd.DataFrame()
    detr_df_raw = pd.DataFrame()
    yolov8_summary = ""
    detr_summary = ""

    if not img_path: # If image processing/download failed
        return original_img_rgb, yolov8_img_np, detr_img_np, \
               yolov8_summary, detr_summary, \
               yolov8_df_raw, detr_df_raw, \
               yolov8_time, detr_time

    # Load original image first to potentially catch UnidentifiedImageError early
    try:
        original_img_pil = Image.open(img_path).convert("RGB")
        original_img_rgb = np.array(original_img_pil)
    except UnidentifiedImageError:
        st.error("Error: Could not identify or open the image. Please ensure it's a valid image file.")
        return original_img_rgb, yolov8_img_np, detr_img_np, \
               yolov8_summary, detr_summary, \
               yolov8_df_raw, detr_df_raw, \
               yolov8_time, detr_time
    except Exception as e:
        st.error(f"Error loading original image: {e}")
        return original_img_rgb, yolov8_img_np, detr_img_np, \
               yolov8_summary, detr_summary, \
               yolov8_df_raw, detr_df_raw, \
               yolov8_time, detr_time

    # Run YOLOv8 model
    try:
        yolov8_person_data, yolov8_time, yolov8_img_np, yolov8_raw_detections = run_yolov8_crowd_detection(yolov8_model, img_path)
        yolov8_df_raw = pd.DataFrame(yolov8_raw_detections)
        yolov8_summary = format_crowd_results("YOLOv8 (Ultralytics)", yolov8_person_data, yolov8_time)
    except Exception as e:
        st.error(f"YOLOv8 analysis failed: {e}")
        # Default values already set at the start

    # Run DETR model
    try:
        detr_person_data, detr_time, detr_img_np, detr_raw_detections = run_detr_hf_crowd_detection(detr_pipeline, img_path)
        detr_df_raw = pd.DataFrame(detr_raw_detections)
        detr_summary = format_crowd_results("DETR (Hugging Face)", detr_person_data, detr_time)
    except Exception as e:
        st.error(f"DETR analysis failed: {e}")
        # Default values already set at the start

    return original_img_rgb, yolov8_img_np, detr_img_np, \
           yolov8_summary, detr_summary, \
           yolov8_df_raw, detr_df_raw, \
           yolov8_time, detr_time

# Streamlit UI
st.title("Crowd Detection Comparison: YOLOv8 (Ultralytics) vs. DETR (Hugging Face)")
st.markdown(
    """
    Upload an image or enter a URL to detect and count people in crowds using both
    YOLOv8 from Ultralytics and DETR from Hugging Face Transformers.
    """
)

# Input section
with st.container():
    col1, col2 = st.columns(2)
    uploaded_file = None
    image_url = ""

    with col1:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    with col2:
        image_url = st.text_input("OR Enter Image URL (optional)", "")

    # Check if any input is provided before proceeding
    if uploaded_file is not None or image_url:
        temp_input_for_analysis = None
        
        if uploaded_file:
            # Streamlit's file_uploader returns a BytesIO object.
            # Save it to a temporary file first.
            temp_image_path = f"uploaded_{uuid.uuid4().hex[:6]}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.read())
            temp_input_for_analysis = temp_image_path
        elif image_url:
            temp_input_for_analysis = image_url
        
        if temp_input_for_analysis: # Only proceed if a valid input was obtained
            with st.spinner("Analyzing image... This may take a moment."):
                original_img_rgb, yolov8_img_np, detr_img_np, \
                yolov8_summary, detr_summary, \
                yolov8_df_raw, detr_df_raw, \
                yolov8_time, detr_time = analyze_image_crowd(image_path_or_url=temp_input_for_analysis)

            if original_img_rgb is not None: # Check if analysis was successful (at least original image loaded)
                st.markdown("---")
                st.header("Results")

                # Display images
                img_cols = st.columns(3)
                with img_cols[0]:
                    st.image(original_img_rgb, caption="Original Image", use_column_width=True)
                with img_cols[1]:
                    st.image(yolov8_img_np if yolov8_img_np is not None else original_img_rgb, caption="YOLOv8 Detections", use_column_width=True)
                with img_cols[2]:
                    st.image(detr_img_np if detr_img_np is not None else original_img_rgb, caption="DETR Detections", use_column_width=True)

                st.markdown("---")
                st.header("Comparison Summary")
                st.markdown("Here's a quick overview of the crowd detection results from both models:")

                # Create two columns for side-by-side display of the comparison table and detailed summaries
                comp_table_col, detailed_summary_col = st.columns(2)

                with comp_table_col:
                    st.subheader("Performance Metrics")
                    yolov8_summary_data_for_table = {PERSON_LABEL: {'count': yolov8_df_raw.shape[0] if not yolov8_df_raw.empty else 0,
                                                                   'confidences': yolov8_df_raw['score'].tolist() if not yolov8_df_raw.empty else [],
                                                                   'true': yolov8_df_raw[yolov8_df_raw['valid'] == True].shape[0] if not yolov8_df_raw.empty else 0,
                                                                   'false': yolov8_df_raw[yolov8_df_raw['valid'] == False].shape[0] if not yolov8_df_raw.empty else 0}}

                    detr_summary_data_for_table = {PERSON_LABEL: {'count': detr_df_raw.shape[0] if not detr_df_raw.empty else 0,
                                                                   'confidences': detr_df_raw['score'].tolist() if not detr_df_raw.empty else [],
                                                                   'true': detr_df_raw[detr_df_raw['valid'] == True].shape[0] if not detr_df_raw.empty else 0,
                                                                   'false': detr_df_raw[detr_df_raw['valid'] == False].shape[0] if not detr_df_raw.empty else 0}}

                    comparison_table = create_comparison_table(yolov8_summary_data_for_table, detr_summary_data_for_table, yolov8_time, detr_time)
                    st.dataframe(comparison_table, hide_index=True)

                with detailed_summary_col:
                    st.subheader("Detailed Summaries")
                    # Use st.text for pre-formatted output and st.markdown for summaries
                    st.text("YOLOv8 (Ultralytics) Summary:")
                    st.markdown(yolov8_summary)
                    st.text("DETR (Hugging Face) Summary:")
                    st.markdown(detr_summary)

            st.markdown("---")
            st.header("Raw Detections Data")
            tab1, tab2 = st.tabs(["YOLOv8 Raw Detections", "DETR Raw Detections"])
            with tab1:
                st.dataframe(yolov8_df_raw)
            with tab2:
                st.dataframe(detr_df_raw)
