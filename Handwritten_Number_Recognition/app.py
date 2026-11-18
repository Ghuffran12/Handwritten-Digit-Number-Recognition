import re
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(
    page_title="Handwritten Number Recognition",
    page_icon="🔢",
    layout="wide"
)

MODEL_NAME = "microsoft/trocr-base-handwritten"

@st.cache_resource(show_spinner="Downloading/loading the handwriting model (first run may take a few minutes)...")
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def crop_to_ink(rgb: np.ndarray, remove_lines: bool = False):
    """Crop around likely handwriting while avoiding aggressive digit segmentation."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold finds dark writing under uneven lighting.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )

    if remove_lines:
        # Optional notebook-line removal. This can damage digits that overlap lines,
        # so it is OFF by default.
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(30, rgb.shape[1] // 4), 1)
        )
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        line_mask = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
        gray = cv2.inpaint(gray, line_mask, 3, cv2.INPAINT_TELEA)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 12
        )

    # Ignore tiny specks when determining the crop, but retain original pixels in crop.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    min_area = max(8, int(rgb.shape[0] * rgb.shape[1] * 0.000015))
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return Image.fromarray(rgb), gray

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    pad = max(12, int(max(x2 - x1, y2 - y1) * 0.10))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(rgb.shape[1], x2 + pad)
    y2 = min(rgb.shape[0], y2 + pad)

    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = rgb
    return Image.fromarray(crop), gray

def recognize(image: Image.Image, processor, model, device, beams: int):
    pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
    pixels = pixels.to(device)
    with torch.inference_mode():
        output = model.generate(
            pixels,
            max_new_tokens=64,
            num_beams=beams,
            early_stopping=True
        )
    text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    digits = re.sub(r"\D", "", text)
    return text, digits

st.title("🔢 Handwritten Number Recognition")
st.write(
    "Upload a photo containing one or more handwritten digits. "
    "The app recognizes the complete number as a sequence; it does not split "
    "the image into individual digits."
)

st.info(
    "Accuracy note: no model can honestly guarantee 100% accuracy or 100% confidence "
    "on every real-world photograph. Results depend on handwriting, image quality, "
    "lighting, spacing and notebook lines. This app uses a pretrained handwriting OCR model."
)

with st.sidebar:
    st.header("Recognition settings")
    remove_lines = st.checkbox(
        "Attempt to remove notebook lines",
        value=False,
        help="Optional. Keep off if the writing overlaps ruled lines; line removal can erase parts of digits."
    )
    beams = st.slider("Recognition search quality (beam size)", 1, 8, 4)
    st.caption("Higher beam size may be slower. It does not guarantee higher accuracy.")

uploaded = st.file_uploader(
    "Choose a photo",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded:
    try:
        original_pil = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        original_rgb = pil_to_rgb(original_pil)

        # Limit extreme image dimensions while preserving aspect ratio.
        max_side = 2200
        if max(original_pil.size) > max_side:
            ratio = max_side / max(original_pil.size)
            original_pil = original_pil.resize(
                (int(original_pil.width * ratio), int(original_pil.height * ratio)),
                Image.Resampling.LANCZOS
            )

        processed_pil, gray = crop_to_ink(
            pil_to_rgb(original_pil), remove_lines=remove_lines
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Uploaded photograph")
            st.image(original_pil, use_container_width=True)
        with right:
            st.subheader("Detected writing region")
            st.image(processed_pil, use_container_width=True)

        if st.button("Recognize handwritten number", type="primary"):
            processor, model, device = load_ocr_model()
            with st.spinner("Recognizing handwriting..."):
                raw_text, number = recognize(
                    processed_pil, processor, model, device, beams
                )

            st.subheader("Recognition result")
            if number:
                st.success(f"Recognized number: **{number}**")
                st.metric("Number of digits", len(number))
            else:
                st.warning(
                    "The model did not return any digits. Try a clearer, closer photo "
                    "with the number centered and all digits visible."
                )

            st.write("Raw handwriting model output:", raw_text or "(empty)")
            st.caption(
                "The model's generated text is shown for transparency. "
                "A calibrated confidence percentage is not provided because this "
                "model's generation score is not a reliable probability of correctness."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
        st.exception(exc)
else:
    st.markdown(
        """
        **Tips for better results**
        - Photograph the complete number, with every digit visible.
        - Keep the camera parallel to the page and use bright, even lighting.
        - Leave a little blank space around the number.
        - Avoid shadows, blur, crossed-out writing and multiple lines of text.
        - For ruled paper, try both settings for notebook-line removal.
        """
    )
