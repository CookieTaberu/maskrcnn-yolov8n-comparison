import streamlit as st
import time
import uuid
import urllib.request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from transformers import pipeline
# PIL (Pillow) for image processing
from PIL import Image, ImageDraw, ImageFont

# Configuration Constants (Defined at the top for global access)
CONF_THRESHOLD = 0.5 # Confidence threshold for displaying detections
PERSON_LABEL = "person" # The label for people in COCO dataset (common for these models)

# Initialize Hugging Face Pipelines (Cached to avoid re-loading on every rerun)
@st.cache_resource
def load_yolos_pipeline():
    """Loads the YOLOS Hugging Face pipeline."""
    return pipeline("object-detection", model="hustvl/yolos-tiny")

@st.cache_resource
def load_detr_pipeline():
    """Loads the DETR Hugging Face pipeline."""
    return pipeline("object-detection", model="facebook/detr-resnet-50")

yolos_pipeline = load_yolos_pipeline()
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

def run_hf_crowd_detection(hf_pipeline, image_path):
    """
    Runs a Hugging Face object detection pipeline on an image,
    and specifically processes detections for 'person' class.
    Returns filtered detection data, time taken, and an annotated image focusing on 'person'.
    """
    # Open image using PIL and ensure it's in RGB mode
    img = Image.open(image_path).convert("RGB")

    start = time.time()
    # Pass PIL Image object to the pipeline. Hugging Face pipelines usually handle PIL input.
    predictions = hf_pipeline(img)
    end = time.time()

    data = {PERSON_LABEL: {"count": 0, "confidences": [], "true": 0, "false": 0}} # Initialize data for 'person'
    annotated_img = img.copy() # Create a copy to draw on
    draw = ImageDraw.Draw(annotated_img)
    try:
        # Try to load a default font for drawing text
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Fallback to default PIL font if arial.ttf is not found (common in some environments)
        font = ImageFont.load_default()

    person_detections = []

    for pred in predictions:
        label = pred['label']
        score = pred['score']

        if label == PERSON_LABEL: # Only process 'person' detections for crowd counting
            box = pred['box']
            x_min, y_min, x_max, y_max = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

            # Draw bounding box using PIL
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2) # Green color, 2px thickness

            # Put label and confidence using PIL
            text = f"{label}: {score:.2f}"
            # Adjust text position to ensure it's visible if box is too close to top
            text_y = y_min - 15 if y_min - 15 > 0 else y_min + 5
            draw.text((x_min, text_y), text, fill=(0, 255, 0), font=font)

            conf = float(score)
            valid = conf > CONF_THRESHOLD

            data[PERSON_LABEL]["count"] += 1
            data[PERSON_LABEL]["confidences"].append(conf)
            data[PERSON_LABEL]["true" if valid else "false"] += 1

            person_detections.append({
                "label": label,
                "score": score,
                "box": box,
                "valid": valid
            })

    # Convert the annotated PIL Image back to a NumPy array for Streamlit display
    return data, end - start, np.array(annotated_img), person_detections

def create_comparison_table(yolos_data, detr_data, yolos_time, detr_time):
    """Creates a consolidated DataFrame for model comparison."""
    # Ensure 'person' data exists for both, otherwise default to 0
    yolos_person_count = yolos_data.get(PERSON_LABEL, {}).get("count", 0)
    detr_person_count = detr_data.get(PERSON_LABEL, {}).get("count", 0)

    yolos_avg_conf = yolos_data.get(PERSON_LABEL, {}).get("confidences", [])
    yolos_avg_conf = sum(yolos_avg_conf) / len(yolos_avg_conf) if yolos_avg_conf else 0

    detr_avg_conf = detr_data.get(PERSON_LABEL, {}).get("confidences", [])
    detr_avg_conf = sum(detr_avg_conf) / len(detr_avg_conf) if detr_avg_conf else 0

    comparison_df = pd.DataFrame({
        "Metric": ["Total People Detected", "Average Confidence", "Processing Time (s)"],
        "YOLOS (Hugging Face)": [yolos_person_count, f"{yolos_avg_conf:.2f}", f"{yolos_time:.2f}"],
        "DETR (Hugging Face)": [detr_person_count, f"{detr_avg_conf:.2f}", f"{detr_time:.2f}"]
    })
    return comparison_df


# Streamlit UI
st.set_page_config(layout="wide", page_title="Crowd Detection Comparison")

st.title("Crowd Detection Comparison: YOLOS vs. DETR (Hugging Face)")
st.markdown(
    """
    Upload an image or enter a URL to detect and count people in crowds using both
    YOLOS and DETR models from Hugging Face Transformers.
    """
)

# Input section
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    with col2:
        image_url = st.text_input("OR Enter Image URL (optional)", "")

    if uploaded_file is not None or image_url:
        temp_image_path = None
        if uploaded_file:
            temp_image_path = f"uploaded_{uuid.uuid4().hex[:6]}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif image_url:
            temp_image_path = download_image(image_url)

        if temp_image_path:
            with st.spinner("Analyzing image... This may take a moment."):
                original_img_rgb, yolos_img_np, detr_img_np, \
                yolos_summary, detr_summary, \
                yolos_df_raw, detr_df_raw, \
                yolos_time, detr_time = analyze_image_crowd(input_img=temp_image_path)

            st.markdown("---")
            st.header("Results")

            # Display images
            img_cols = st.columns(3)
            with img_cols[0]:
                st.image(original_img_rgb, caption="Original Image", use_column_width=True)
            with img_cols[1]:
                st.image(yolos_img_np, caption="YOLOS Detections", use_column_width=True)
            with img_cols[2]:
                st.image(detr_img_np, caption="DETR Detections", use_column_width=True)

            st.markdown("---")
            st.header("Comparison Summary")
            st.markdown("Here's a quick overview of the crowd detection results from both models:")
            comparison_table = create_comparison_table(
                yolos_person_data={PERSON_LABEL: yolos_df_raw.to_dict('list')} if not yolos_df_raw.empty else {},
                detr_person_data={PERSON_LABEL: detr_df_raw.to_dict('list')} if not detr_df_raw.empty else {},
                yolos_time=yolos_time,
                detr_time=detr_time
            )
            # Re-process data for comparison table to extract actual counts/confidences from dataframes
            # Assuming format_crowd_results 'data' param expects a dict with 'person' key directly containing count, confidences, etc.
            # Let's adjust create_comparison_table to properly use the summarized `data` dictionaries.
            yolos_summary_data = {PERSON_LABEL: {'count': yolos_df_raw['label'].count() if not yolos_df_raw.empty else 0,
                                                 'confidences': yolos_df_raw['score'].tolist() if not yolos_df_raw.empty else [],
                                                 'true': yolos_df_raw[yolos_df_raw['valid'] == True].shape[0] if not yolos_df_raw.empty else 0,
                                                 'false': yolos_df_raw[yolos_df_raw['valid'] == False].shape[0] if not yolos_df_raw.empty else 0}}

            detr_summary_data = {PERSON_LABEL: {'count': detr_df_raw['label'].count() if not detr_df_raw.empty else 0,
                                                 'confidences': detr_df_raw['score'].tolist() if not detr_df_raw.empty else [],
                                                 'true': detr_df_raw[detr_df_raw['valid'] == True].shape[0] if not detr_df_raw.empty else 0,
                                                 'false': detr_df_raw[detr_df_raw['valid'] == False].shape[0] if not detr_df_raw.empty else 0}}

            comparison_table = create_comparison_table(yolos_summary_data, detr_summary_data, yolos_time, detr_time)
            st.dataframe(comparison_table, hide_index=True)


            st.markdown("---")
            st.header("Detailed Summaries")
            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.subheader("YOLOS (Hugging Face) Summary")
                st.markdown(yolos_summary)
            with summary_cols[1]:
                st.subheader("DETR (Hugging Face) Summary")
                st.markdown(detr_summary)

            st.markdown("---")
            st.header("Raw Detections Data")
            tab1, tab2 = st.tabs(["YOLOS Raw Detections", "DETR Raw Detections"])
            with tab1:
                st.dataframe(yolos_df_raw)
            with tab2:
                st.dataframe(detr_df_raw)
        else:
            st.info("Please upload an image or provide a URL to begin crowd detection.")

