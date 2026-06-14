from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class TokenGen(nn.Module):
	"""Generate a learnable token bank and concatenate it with features.

	The module is intentionally lightweight: it stores ``num_tokens`` learnable
	vectors and concatenates them with an input tensor along the sequence
	dimension. If the feature dimension is not known at construction time, the
	token table is materialized lazily on the first forward pass.
	"""

	def __init__(
		self,
		num_tokens: int,
		token_dim: int | None = None,
		*,
		concat_position: Literal["prefix", "suffix"] = "prefix",
		init_std: float = 0.02,
	):
		super().__init__()

		if num_tokens < 0:
			raise ValueError("num_tokens must be non-negative")
		if concat_position not in {"prefix", "suffix"}:
			raise ValueError("concat_position must be 'prefix' or 'suffix'")

		self.num_tokens = num_tokens
		self.token_dim = token_dim
		self.concat_position = concat_position
		self.init_std = init_std

		if token_dim is None:
			self.tokens = nn.Parameter(torch.empty(num_tokens, 0))
		else:
			self.tokens = nn.Parameter(torch.empty(num_tokens, token_dim))
			self.reset_parameters()

	def reset_parameters(self):
		if self.tokens.numel() > 0:
			nn.init.normal_(self.tokens, std=self.init_std)

	def _ensure_token_dim(self, feature_dim: int, device: torch.device, dtype: torch.dtype):
		current_dim = self.tokens.shape[-1]
		if current_dim == feature_dim and self.tokens.device == device and self.tokens.dtype == dtype:
			return

		self.token_dim = feature_dim
		self.tokens = nn.Parameter(torch.empty(self.num_tokens, feature_dim, device=device, dtype=dtype))
		self.reset_parameters()

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		if features.dim() not in (2, 3):
			raise ValueError("features must have shape [B, D] or [B, N, D]")

		if self.num_tokens == 0:
			return features if features.dim() == 3 else features.unsqueeze(1)

		if features.dim() == 2:
			features = features.unsqueeze(1)

		self._ensure_token_dim(features.shape[-1], features.device, features.dtype)

		tokens = self.tokens.unsqueeze(0).expand(features.shape[0], -1, -1)
		if self.concat_position == "prefix":
			return torch.cat((tokens, features), dim=1)
		return torch.cat((features, tokens), dim=1)


class TokenizedVisionEncoder(nn.Module):
	"""Run a CLIP vision transformer with token injection at two depths."""

	def __init__(
		self,
		visual: nn.Module,
		low_tokens: TokenGen,
		high_tokens: TokenGen,
		split_index: int | None = None,
	):
		super().__init__()
		self.visual = visual
		self.low_tokens = low_tokens
		self.high_tokens = high_tokens
		self.split_index = split_index

	def _run_blocks(self, x: torch.Tensor, blocks) -> torch.Tensor:
		for block in blocks:
			x = block(x)
		return x

	def _encode_transformer(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
		visual = self.visual
		x = image.type(visual.conv1.weight.dtype)
		x = visual.conv1(x)
		x = x.reshape(x.shape[0], x.shape[1], -1)
		x = x.permute(0, 2, 1)

		class_embedding = visual.class_embedding.to(x.dtype)
		class_token = class_embedding + torch.zeros(
			x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
		)
		x = torch.cat([class_token, x], dim=1)
		x = x + visual.positional_embedding.to(x.dtype)
		x = visual.ln_pre(x)

		blocks = visual.transformer.resblocks
		midpoint = self.split_index if self.split_index is not None else len(blocks) // 2

		x = self.low_tokens(x)
		x = self._run_blocks(x, blocks[:midpoint])
		low_token_features = x[:, 1 : 1 + self.low_tokens.num_tokens, :]
		x = torch.cat((x[:, :1, :], x[:, 1 + self.low_tokens.num_tokens :, :]), dim=1)

		x = self.high_tokens(x)
		x = self._run_blocks(x, blocks[midpoint:])
		high_token_features = x[:, 1 : 1 + self.high_tokens.num_tokens, :]
		x = torch.cat((x[:, :1, :], x[:, 1 + self.high_tokens.num_tokens :, :]), dim=1)

		image_features = visual.ln_post(x[:, 0, :])
		if visual.proj is not None:
			image_features = image_features @ visual.proj

		return {
			"image_features": image_features,
			"low_token_features": low_token_features,
			"high_token_features": high_token_features,
		}

	def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
		if hasattr(self.visual, "transformer") and hasattr(self.visual, "class_embedding"):
			return self._encode_transformer(image)

		image_features = self.visual(image)
		return {
			"image_features": image_features,
			"low_token_features": image_features.new_zeros(image_features.shape[0], 0, image_features.shape[-1]),
			"high_token_features": image_features.new_zeros(image_features.shape[0], 0, image_features.shape[-1]),
		}