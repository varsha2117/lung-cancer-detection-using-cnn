import io
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

from utils.model import load_model, predict_proba
from utils.preprocess import load_image_or_dicom, preprocess_image


st.set_page_config(page_title="CT Lung Cancer Detector", page_icon="🫁", layout="centered")


def main() -> None:
	st.title("🫁 CT Lung Cancer Detection")
	st.write(
		"Upload a chest CT slice or DICOM file. A compact CNN will estimate the probability of lung cancer from the image."
	)

	with st.expander("Model options", expanded=False):
		default_weights = "model_weights.pth"
		weights_path = st.text_input(
			"Weights file (optional)", value=default_weights, help="Path to a PyTorch .pth file"
		)

	uploaded = st.file_uploader(
		"Upload CT image (PNG/JPG) or DICOM (.dcm)",
		type=["png", "jpg", "jpeg", "dcm"],
		accept_multiple_files=False,
	)

	if uploaded is None:
		st.info("Please upload a CT scan to begin.")
		return

	# Load image or DICOM as a PIL image in grayscale
	try:
		pil_image = load_image_or_dicom(uploaded)
	except Exception as e:
		st.error(f"Failed to read file: {e}")
		return

	st.image(pil_image, caption="Input slice", use_column_width=True)

	# Preprocess for the model
	input_tensor = preprocess_image(pil_image)

	# Load or create model
	model = load_model(weights_path if os.path.exists(weights_path) else None)

	# Inference
	prob = predict_proba(model, input_tensor)
	st.subheader("Prediction")
	st.metric("Cancer probability", f"{prob*100:.1f}%")

	st.caption(
		"This tool is not a medical device and is not intended for diagnosis or treatment. Consult qualified professionals for clinical decisions."
	)


if __name__ == "__main__":
	main()


