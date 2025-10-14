from typing import IO

import io
import numpy as np
from PIL import Image
import torch


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
	arr = arr.astype(np.float32)
	min_v = float(arr.min())
	max_v = float(arr.max())
	if max_v - min_v < 1e-6:
		return np.zeros_like(arr, dtype=np.uint8)
	arr = (arr - min_v) / (max_v - min_v)
	arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
	return arr


def load_image_or_dicom(file_like: IO[bytes]) -> Image.Image:
	name = getattr(file_like, "name", "uploaded")
	if str(name).lower().endswith(".dcm"):
		try:
			import pydicom  # type: ignore
		except Exception as e:
			raise RuntimeError("pydicom is required to read DICOM files. Add it to requirements.") from e
		file_bytes = file_like.read()
		ds = pydicom.dcmread(io.BytesIO(file_bytes))
		arr = ds.pixel_array.astype(np.float32)
		arr = _rescale_to_uint8(arr)
		return Image.fromarray(arr).convert("L")
	else:
		return Image.open(file_like).convert("L")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
	# Resize to 224x224, normalize to mean 0.5 std 0.5, add batch/channel dims
	img = pil_img.resize((224, 224))
	arr = np.array(img, dtype=np.float32) / 255.0
	arr = (arr - 0.5) / 0.5
	arr = arr[np.newaxis, np.newaxis, :, :]
	return torch.from_numpy(arr.astype(np.float32))


