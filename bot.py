#!/usr/bin/env python3
# name=bot.py
"""
Fixed bot.py — robust processing, correct download extensions, codec checks,
video_note square-cropping, image-sequence fallback, applied-effect detection,
and reliable sending (InputFile).
Requirements: python-telegram-bot v20+, ffmpeg/ffprobe in PATH, OpenCV, MediaPipe, Pillow, numpy.
"""
import os
import logging
import asyncio
import subprocess
import tempfile
import threading
import shutil
import mimetypes
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.error import Conflict as TGConflict
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ---------- CONFIG ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # optional
PORT = int(os.environ.get("PORT", 8443))
if not BOT_TOKEN:
    raise SystemExit("Error: BOT_TOKEN env var is not set")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands

INDEX_TIP = 8
THUMB_TIP = 4

# Colors/text UI
COLORS = {
    "purple": {"name": "🟣 Пурпурный", "bgr": (255, 0, 200)},
    "blue": {"name": "🔵 Синий", "bgr": (255, 100, 0)},
    "green": {"name": "🟢 Зелёный", "bgr": (60, 255, 60)},
    "red": {"name": "🔴 Красный", "bgr": (40, 30, 230)},
    "yellow": {"name": "🟡 Жёлтый", "bgr": (40, 230, 230)},
    "white": {"name": "⚪ Белый", "bgr": (240, 240, 240)},
}

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def find_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


FONT_PATH = find_font()

# ---------- utility (ffprobe helpers, safe temp names) ----------


def ffprobe_duration(path):
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        if p.returncode == 0 and p.stdout.strip():
            return float(p.stdout.strip())
    except Exception:
        pass
    return None


def ffprobe_video_codec(path):
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return None


def get_safe_tmp_path(basename="in", ext=".mp4"):
    tmp = tempfile.NamedTemporaryFile(prefix=basename + "_", suffix=ext, delete=False)
    tmp.close()
    return tmp.name


def extension_from_mime(mime):
    if not mime:
        return ".mp4"
    ext = mimetypes.guess_extension(mime.split(";")[0].strip())
    if ext:
        return ext
    # fallback mapping
    if "webm" in mime:
        return ".webm"
    if "mp4" in mime:
        return ".mp4"
    return ".mp4"


# ---------- hand geometry and detection ----------


def _finger_extended(lm, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
    wrist = np.array([lm[wrist_idx].x, lm[wrist_idx].y])
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y])
    mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y])
    dist_tip = np.linalg.norm(tip - wrist)
    dist_pip = np.linalg.norm(pip - wrist)
    dist_mcp = np.linalg.norm(mcp - wrist)
    return dist_tip > dist_pip * 1.05 and dist_tip > dist_mcp * 1.15


def get_hand_points(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    if not _finger_extended(lm, 8, 6, 5):
        return None
    wrist = np.array([lm[0].x, lm[0].y])
    index_tip = np.array([lm[8].x, lm[8].y])
    dist_index = np.linalg.norm(index_tip - wrist)
    other_tips = [12, 16, 20]
    curled_count = 0
    for tip_idx in other_tips:
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
        dist_tip = np.linalg.norm(tip - wrist)
        if dist_tip < dist_index * 0.75:
            curled_count += 1
    if curled_count < 2:
        return None
    thumb_tip = np.array([lm[THUMB_TIP].x, lm[THUMB_TIP].y])
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    if thumb_index_dist < dist_index * 0.3:
        return None
    ix = int(lm[INDEX_TIP].x * w)
    iy = int(lm[INDEX_TIP].y * h)
    tx = int(lm[THUMB_TIP].x * w)
    ty = int(lm[THUMB_TIP].y * h)
    return (ix, iy), (tx, ty)


# ---------- drawing/effects ----------


def draw_unicode_glow_text(img_bgr, text, center, box_width, color_bgr):
    if not FONT_PATH:
        return img_bgr
    h, w = img_bgr.shape[:2]
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    font_size = max(14, min(60, int(box_width / (len(text) * 0.62 + 1))))
    font = ImageFont.truetype(FONT_PATH, font_size)
    glow_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_glow = ImageDraw.Draw(glow_img)
    bbox = draw_glow.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = center[0] - tw / 2 - bbox[0]
    y = center[1] - th / 2 - bbox[1]
    draw_glow.text((x, y), text, font=font, fill=(*color_rgb, 255))
    glow_np = cv2.cvtColor(np.array(glow_img), cv2.COLOR_RGBA2BGRA)
    glow_blur = cv2.GaussianBlur(glow_np, (9, 9), 0)
    glow_bgr = glow_blur[:, :, :3]
    glow_alpha = (glow_blur[:, :, 3:4].astype(np.float32) / 255.0) * 0.5
    img_bgr[:] = (img_bgr.astype(np.float32) * (1 - glow_alpha) + glow_bgr.astype(np.float32) * glow_alpha).astype(np.uint8)
    sharp_np = cv2.cvtColor(np.array(glow_img), cv2.COLOR_RGBA2BGRA)
    sharp_bgr = sharp_np[:, :, :3]
    sharp_alpha = sharp_np[:, :, 3:4].astype(np.float32) / 255.0
    img_bgr[:] = (img_bgr.astype(np.float32) * (1 - sharp_alpha) + sharp_bgr.astype(np.float32) * sharp_alpha).astype(np.uint8)
    return img_bgr


def apply_quad_effect(frame, pts, text=None, color_bgr=(255, 0, 200)):
    H, W = frame.shape[:2]
    x1 = max(0, int(pts[:, 0].min()) - 2)
    y1 = max(0, int(pts[:, 1].min()) - 2)
    x2 = min(W, int(pts[:, 0].max()) + 2)
    y2 = min(H, int(pts[:, 1].max()) + 2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    pts_local = pts.copy()
    pts_local[:, 0] -= x1
    pts_local[:, 1] -= y1
    pts_local = pts_local.astype(np.int32)
    pts_local[:, 0] = np.clip(pts_local[:, 0], 0, x2 - x1 - 1)
    pts_local[:, 1] = np.clip(pts_local[:, 1], 0, y2 - y1 - 1)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_local], 255)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    dark = (roi * 0.06).astype(np.uint8)
    inv = cv2.bitwise_not(roi)
    gray_inv = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    tinted = np.zeros_like(roi)
    for i in range(3):
        tinted[:, :, i] = (gray_inv.astype(np.float32) * (color_bgr[i] / 255.0)).astype(np.uint8)
    glow = cv2.GaussianBlur(tinted, (21, 21), 0)
    effect = cv2.addWeighted(dark, 0.15, tinted, 0.55, 0)
    effect = cv2.addWeighted(effect, 1.0, glow, 0.45, 0)
    if text:
        cx = int(pts_local[:, 0].mean())
        cy = int(pts_local[:, 1].mean())
        quad_w = int(max(pts_local[:, 0]) - min(pts_local[:, 0]))
        effect = draw_unicode_glow_text(effect, text, (cx, cy), quad_w, color_bgr)
    roi_result = (roi.astype(np.float32) * (1 - mask3) + effect.astype(np.float32) * mask3).astype(np.uint8)
    glow_layer = roi_result.copy()
    cv2.polylines(glow_layer, [pts_local], True, color_bgr, max(2, int(round(min(W, H) * 0.02))))
    roi_result = cv2.addWeighted(roi_result, 0.72, glow_layer, 0.28, 0)
    cv2.polylines(roi_result, [pts_local], True, color_bgr, 2)
    frame[y1:y2, x1:x2] = roi_result
    return frame


# ---------- detection pipeline (same logic as before but with applied flag) ----------


def detect_hands_and_apply(hands_detector, frame, w, h, prev_pts, prev_center, smooth_base, smooth_min, jump_quick, MAX_JUMP, area_ratio_min, area_ratio_max):
    """
    Returns: (frame_out, new_prev_pts, new_prev_center, applied_this_frame(bool), new_prev_area or None)
    """
    raw_points = detect_hands(hands_detector, frame, w, h)
    raw_points = dedupe_hands(raw_points)
    pair = pick_best_pair(raw_points, prev_center)
    if pair is None:
        return frame, prev_pts, prev_center, False, None, None

    try:
        quad = make_quad(pair[0], pair[1])
        area = cv2.contourArea(quad)
        new_center = quad.mean(axis=0)
        if area < 150:
            return frame, prev_pts, prev_center, False, None, None

        if prev_center is None or prev_pts is None:
            prev_pts = quad.astype(np.float32)
            prev*

