from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.25)
		self.fc1 = nn.Linear(32 * 56 * 56, 64)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(F.relu(self.conv1(x)))  # 1->16, 224->112
		x = self.pool(F.relu(self.conv2(x)))  # 16->32, 112->56
		x = self.dropout(x)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


def load_model(weights_path: Optional[str]) -> nn.Module:
	model = SimpleCNN()
	model.eval()
	if weights_path:
		try:
			state = torch.load(weights_path, map_location="cpu")
			model.load_state_dict(state)
		except Exception:
			# If loading fails, continue with randomly initialized weights
			pass
	return model


@torch.inference_mode()
def predict_proba(model: nn.Module, input_tensor: torch.Tensor) -> float:
	logits = model(input_tensor)  # shape [1, 1]
	proba = torch.sigmoid(logits).item()
	return float(proba)


