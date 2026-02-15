"""Fisheye corner alpha packer for DeoVR/HereSphere passthrough.

Packs the RVM alpha matte into the 8 corner dead zones of SBS fisheye
frames using the red channel, enabling AR passthrough on Quest 3S.

The fisheye circle occupies the center of each eye, leaving rectangular
dead zones in all four corners. The alpha matte is downscaled, split
into 4 quadrants, and painted into each corner via the red channel.
Both eyes get identical alpha data (matte is computed from one eye).

Dependencies: NumPy + OpenCV only (no PyTorch).

Corner layout per eye (e.g., 4000x4000 eye → 1600x1600 scaled matte):
┌─Q1────┐          ┌─Q2────┐
│ red   │  fisheye │ red   │
└───────┘  circle  └───────┘

┌─Q3────┐          ┌─Q4────┐
│ red   │  fisheye │ red   │
└───────┘  circle  └───────┘
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ppp.alpha_packer")


class AlphaPacker:
    """Packs alpha matte into fisheye corner dead zones."""

    def __init__(self, scale: float = 0.4):
        """
        Args:
            scale: Downscale factor for the matte relative to eye dimensions.
                   0.4 means 40% of eye width/height → 4 quadrants each 20%.
        """
        self.scale = scale

    def pack(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        eye_width: Optional[int] = None,
    ) -> np.ndarray:
        """Pack alpha matte into the corner dead zones of an SBS frame.

        Modifies frame IN-PLACE for zero-copy performance and also returns it.

        Args:
            frame: Full SBS frame [H, W, 3] uint8 (W = 2 * eye_width)
            alpha: Single-eye alpha matte [eye_H, eye_W] float32 (0.0-1.0)
            eye_width: Width of a single eye. Defaults to frame.shape[1] // 2.

        Returns:
            The modified frame (same object, modified in-place).
        """
        h, full_w = frame.shape[:2]
        if eye_width is None:
            eye_width = full_w // 2

        # Downscale matte to target size
        scaled_h = int(h * self.scale)
        scaled_w = int(eye_width * self.scale)
        scaled_alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Convert to uint8 for red channel painting
        alpha_u8 = (np.clip(scaled_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Split into 4 quadrants
        half_h = scaled_h // 2
        half_w = scaled_w // 2
        q1 = alpha_u8[:half_h, :half_w]         # top-left
        q2 = alpha_u8[:half_h, half_w:scaled_w]  # top-right
        q3 = alpha_u8[half_h:scaled_h, :half_w]  # bottom-left
        q4 = alpha_u8[half_h:scaled_h, half_w:scaled_w]  # bottom-right

        # Paint into both eyes
        for eye_offset in (0, eye_width):
            self._paint_corner(frame, q1, eye_offset, 0, 0)                           # top-left
            self._paint_corner(frame, q2, eye_offset, eye_width - half_w, 0)           # top-right
            self._paint_corner(frame, q3, eye_offset, 0, h - half_h)                   # bottom-left
            self._paint_corner(frame, q4, eye_offset, eye_width - half_w, h - half_h)  # bottom-right

        return frame

    @staticmethod
    def _paint_corner(
        frame: np.ndarray,
        quadrant: np.ndarray,
        eye_offset: int,
        x: int,
        y: int,
    ):
        """Paint a quadrant into the red channel at the specified position.

        Args:
            frame: Full SBS frame [H, W, 3] uint8 (RGB order)
            quadrant: Alpha quadrant [qh, qw] uint8
            eye_offset: Horizontal offset for the eye (0 or eye_width)
            x: X position within the eye
            y: Y position within the frame
        """
        qh, qw = quadrant.shape
        abs_x = eye_offset + x
        # Paint red channel only — leaves green and blue untouched
        frame[y:y + qh, abs_x:abs_x + qw, 0] = quadrant
