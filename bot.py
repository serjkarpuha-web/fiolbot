#!/usr/bin/env python3
# name=bot.py
import os
import logging
import asyncio
import subprocess
import tempfile
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.error import Conflict as TGConflict
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
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

# Available colors
COLORS = {
    "purple": {"name": "🟣 Пурпурный", "bgr": (255, 0, 200)},
    "blue":   {"name": "🔵 Синий",     "bgr": (255, 100, 0)},
    "green":  {"name": "🟢 Зелёный",   "bgr": (60, 255, 60)},
    "red":    {"name": "🔴 Красный",   "bgr": (40, 30, 230)},
    "yellow": {"name": "🟡 Жёлтый",    "bgr": (40, 230, 230)},
    "white":  {"name": "⚪ Белый",     "bgr": (240, 240, 240)},
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

# ---------- Geometry helpers ----------


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
    index_extended = _finger_extended(lm, 8, 6, 5)
    if not index_extended:
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


# ---------- Rendering helpers ----------


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


# ---------- Detection helpers ----------


def detect_hands(hands_detector, frame, w, h):
    MAX_SIDE = 640
    scale = 1.0
    search_frame = frame
    longest_side = max(w, h)
    if longest_side > MAX_SIDE:
        scale = MAX_SIDE / longest_side
        sw, sh = int(w * scale), int(h * scale)
        search_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(search_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    try:
        results = hands_detector.process(rgb)
    except Exception as e:
        logger.warning("MediaPipe failed on frame, skipping: %s", e)
        return []
    found = []
    if results.multi_hand_landmarks:
        sh_, sw_ = search_frame.shape[:2]
        for hand_lm in results.multi_hand_landmarks:
            pts = get_hand_points(hand_lm, sw_, sh_)
            if pts is not None:
                idx_pt, thm_pt = pts
                idx_orig = (int(idx_pt[0] / scale), int(idx_pt[1] / scale))
                thm_orig = (int(thm_pt[0] / scale), int(thm_pt[1] / scale))
                found.append((idx_orig, thm_orig))
    return found


def dedupe_hands(hand_points, dist_threshold=60):
    unique = []
    for idx_pt, thm_pt in hand_points:
        is_dup = False
        for u_idx, u_thm in unique:
            if (abs(idx_pt[0] - u_idx[0]) < dist_threshold and abs(idx_pt[1] - u_idx[1]) < dist_threshold):
                is_dup = True
                break
        if not is_dup:
            unique.append((idx_pt, thm_pt))
    return unique


def make_quad(hand_a, hand_b):
    a_idx, a_thm = hand_a
    b_idx, b_thm = hand_b
    a_cx = (a_idx[0] + a_thm[0]) // 2
    b_cx = (b_idx[0] + b_thm[0]) // 2
    if a_cx > b_cx:
        a_idx, a_thm, b_idx, b_thm = b_idx, b_thm, a_idx, a_thm
    tl = np.array(a_idx)
    tr = np.array(b_idx)
    br = np.array(b_thm)
    bl = np.array(a_thm)
    return np.array([tl, tr, br, bl], dtype=np.int32)


def pick_best_pair(hand_points, prev_center=None):
    if len(hand_points) < 2:
        return None
    if len(hand_points) == 2:
        return hand_points[0], hand_points[1]
    from itertools import combinations
    candidates = []
    for a, b in combinations(hand_points, 2):
        try:
            quad = make_quad(a, b)
            area = cv2.contourArea(quad)
            if area < 150:
                continue
            center = quad.mean(axis=0)
            candidates.append((area, center, a, b))
        except Exception:
            continue
    if not candidates:
        return None
    if prev_center is not None:
        candidates.sort(key=lambda c: np.linalg.norm(c[1] - prev_center))
    else:
        candidates.sort(key=lambda c: -c[0])
    _, _, a, b = candidates[0]
    return a, b


# ---------- ffmpeg helpers ----------


def ffprobe_duration(path):
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        if p.returncode == 0 and p.stdout.strip():
            return float(p.stdout.strip())
    except Exception:
        pass
    return None


def prepare_work_input(input_path, max_seconds=20):
    """
    Convert / trim input to mp4/h264 (mp4) up to max_seconds.
    Returns (work_input_path, created_tmp_bool)
    """
    duration = ffprobe_duration(input_path)
    need_trim = False
    if duration is not None and duration > max_seconds + 0.01:
        need_trim = True
    ext = os.path.splitext(input_path)[1].lower()
    need_convert = ext not in (".mp4", ".mov", ".m4v") or need_trim
    if not need_convert:
        return input_path, False
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-ss", "0", "-t", str(max_seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", out_tmp.name
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.exists(out_tmp.name) and os.path.getsize(out_tmp.name) > 1000:
            return out_tmp.name, True
        else:
            logger.warning("ffmpeg convert failed, fallback to input. stderr: %s", r.stderr.decode(errors="ignore")[-400:])
            try:
                os.unlink(out_tmp.name)
            except Exception:
                pass
            return input_path, False
    except Exception as e:
        logger.warning("ffmpeg convert exception: %s", e)
        try:
            os.unlink(out_tmp.name)
        except Exception:
            pass
        return input_path, False


# ---------- CORE VIDEO PROCESSING with robust fallback to image-sequence ----------


def _process_video_inner(input_path: str, output_path: str, custom_text=None, color_bgr=(255, 0, 200), cancel_event: threading.Event = None):
    work_input, created_tmp = prepare_work_input(input_path, max_seconds=20)

    cap = cv2.VideoCapture(work_input)
    if not cap.isOpened():
        logger.error("cv2 cannot open video %s", work_input)
        if created_tmp and os.path.exists(work_input):
            os.remove(work_input)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1.0:
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = ffprobe_duration(work_input)
            if dur and frame_count and dur > 0:
                fps = max(1.0, frame_count / dur)
            else:
                fps = 25.0
        except Exception:
            fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # try VideoWriter first (fast), fallback to image-sequence+ffmpeg if it fails
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_path = tmp_video.name
    tmp_video.close()

    fourcc_candidates = ["mp4v", "X264", "avc1", "H264"]
    out = None
    for fc in fourcc_candidates:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fc)
            out = cv2.VideoWriter(tmp_video_path, fourcc, float(fps), (w, h))
            if out.isOpened():
                logger.info("Opened VideoWriter with fourcc=%s fps=%.2f size=%dx%d -> %s", fc, fps, w, h, tmp_video_path)
                break
            else:
                try:
                    out.release()
                except Exception:
                    pass
                out = None
        except Exception:
            out = None

    use_image_seq = out is None or not out.isOpened()
    frames_dir = None
    frame_idx = 0
    if use_image_seq:
        frames_dir = tempfile.mkdtemp(prefix="frames_")
        logger.info("VideoWriter unavailable, will write frames to %s and assemble with ffmpeg", frames_dir)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0)

    prev_pts = None
    prev_center = None
    prev_area = None
    pts_history = deque(maxlen=5)
    area_history = deque(maxlen=5)
    outlier_count = 0

    smooth_base = 0.8
    smooth_min = 0.2
    jump_quick = max(w, h) * 0.12
    MAX_JUMP = max(w, h) * 0.9
    area_ratio_min = 0.5
    area_ratio_max = 2.0
    miss_count = 0
    MAX_MISS = 6

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info("Start processing: %dx%d, ~%d frames, fps=%.2f", w, h, total_frames, fps)

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Processing canceled by user")
                return False

            ret, frame = cap.read()
            if not ret:
                break

            try:
                raw_points = detect_hands(hands, frame, w, h)
                raw_points = dedupe_hands(raw_points)
                pair = pick_best_pair(raw_points, prev_center)

                valid_update = False
                if pair is not None:
                    try:
                        quad = make_quad(pair[0], pair[1])
                        area = cv2.contourArea(quad)
                        new_center = quad.mean(axis=0)
                        if area < 150:
                            raise ValueError("area too small")
                        if prev_center is None or prev_pts is None:
                            prev_pts = quad.astype(np.float32)
                            prev_center = prev_pts.mean(axis=0)
                            prev_area = area
                            pts_history.clear()
                            area_history.clear()
                            pts_history.append(prev_pts.copy())
                            area_history.append(prev_area)
                            frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                            miss_count = 0
                            valid_update = True
                        else:
                            jump = np.linalg.norm(new_center - prev_center)
                            area_ratio = area / (prev_area + 1e-6)
                            if jump > MAX_JUMP or not (area_ratio_min <= area_ratio <= area_ratio_max):
                                outlier_count += 1
                                if outlier_count >= 3:
                                    alpha = 1.0 - smooth_min
                                    prev_pts = prev_pts * (1 - alpha) + quad.astype(np.float32) * alpha
                                    prev_center = prev_pts.mean(axis=0)
                                    prev_area = prev_area * 0.7 + area * 0.3
                                    pts_history.append(prev_pts.copy())
                                    area_history.append(prev_area)
                                    frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                                    miss_count = 0
                                    valid_update = True
                            else:
                                outlier_count = 0
                                pts_history.append(quad.astype(np.float32))
                                area_history.append(area)
                                if jump > jump_quick:
                                    alpha = 1.0 - smooth_min
                                else:
                                    alpha = 1.0 - smooth_base
                                prev_pts = prev_pts * (1 - alpha) + quad.astype(np.float32) * alpha
                                prev_center = prev_pts.mean(axis=0)
                                prev_area = prev_area * 0.85 + area * 0.15
                                frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                                miss_count = 0
                                valid_update = True
                    except Exception:
                        valid_update = False

                if not valid_update:
                    miss_count += 1
                    if prev_pts is not None and miss_count <= MAX_MISS:
                        frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                    else:
                        prev_pts = None
                        prev_center = None
                        prev_area = None
                        pts_history.clear()
                        area_history.clear()
            except Exception as e:
                logger.warning("Frame %d: unexpected error, writing frame without effect: %s", frame_idx, e)

            # write frame
            if not use_image_seq:
                out.write(frame)
            else:
                # save as PNG (lossless) for ffmpeg assembly
                fname = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(fname, frame)
            frame_idx += 1

            if frame_idx % 60 == 0:
                logger.info("Progress: %d/%d frames", frame_idx, total_frames)

    finally:
        cap.release()
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        hands.close()

    # if used image sequence, assemble with ffmpeg
    if use_image_seq:
        if frame_idx == 0:
            logger.error("No frames rendered")
            # cleanup frames_dir
            try:
                if os.path.exists(frames_dir):
                    import shutil
                    shutil.rmtree(frames_dir)
            except Exception:
                pass
            if created_tmp and os.path.exists(work_input):
                try:
                    os.remove(work_input)
                except Exception:
                    pass
            return False
        # assemble
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp_video_path
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=300)
            if r.returncode != 0:
                logger.error("ffmpeg assemble frames failed: %s", r.stderr.decode(errors="ignore")[-500:])
                # cleanup frames_dir
                import shutil
                try:
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass
                if created_tmp and os.path.exists(work_input):
                    try:
                        os.remove(work_input)
                    except Exception:
                        pass
                return False
        except Exception as e:
            logger.exception("ffmpeg assemble exception: %s", e)
            import shutil
            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass
            if created_tmp and os.path.exists(work_input):
                try:
                    os.remove(work_input)
                except Exception:
                    pass
            return False
        # cleanup frames
        import shutil
        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

    logger.info("Intermediate video created: %s (%d bytes)", tmp_video_path, os.path.getsize(tmp_video_path) if os.path.exists(tmp_video_path) else 0)

    # final merge with original audio (if any)
    cmd = ["ffmpeg", "-y", "-i", tmp_video_path, "-i", input_path, "-c:v", "libx264", "-c:a", "aac",
           "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-pix_fmt", "yuv420p", output_path]
    try:
        r1 = subprocess.run(cmd, capture_output=True, timeout=300)
        if r1.returncode != 0:
            logger.warning("ffmpeg (video+audio) failed, trying video-only. stderr: %s", r1.stderr.decode(errors="ignore")[-400:])
            cmd2 = ["ffmpeg", "-y", "-i", tmp_video_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
            r2 = subprocess.run(cmd2, capture_output=True, timeout=300)
            if r2.returncode != 0:
                logger.error("ffmpeg (video-only) also failed: %s", r2.stderr.decode(errors="ignore")[-400:])
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass
                if created_tmp and os.path.exists(work_input):
                    try:
                        os.remove(work_input)
                    except Exception:
                        pass
                return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass
        if created_tmp and os.path.exists(work_input):
            try:
                os.remove(work_input)
            except Exception:
                pass
        return False

    try:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
    except Exception:
        pass

    if created_tmp and os.path.exists(work_input):
        try:
            os.remove(work_input)
        except Exception:
            pass

    # quick validation via ffprobe
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "csv=p=0", output_path],
            capture_output=True, text=True, timeout=30
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            logger.error("ffprobe cannot validate output: rc=%s stderr=%s", probe.returncode, probe.stderr)
            return False
    except Exception as e:
        logger.error("ffprobe check exception: %s", e)
        return False

    logger.info("Video processed and verified successfully: %s", output_path)
    return True


def process_video(input_path: str, output_path: str, custom_text=None, color_bgr=(255, 0, 200), cancel_event: threading.Event = None):
    import traceback
    try:
        return _process_video_inner(input_path, output_path, custom_text, color_bgr, cancel_event=cancel_event)
    except Exception as e:
        logger.error("CRITICAL process_video error: %s\n%s", e, traceback.format_exc())
        return False


# ---------- Async wrapper + Telegram integration ----------


async def run_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, input_path: str, custom_text=None, color_bgr=(255, 0, 200)):
    msg = update.effective_message
    status_msg = await msg.reply_text("🔄 Получено, скачиваю и обрабатываю видео...")
    output_path = input_path + "_out.mp4"
    cancel_event = threading.Event()
    context.user_data["cancel_event"] = cancel_event

    try:
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, process_video, input_path, output_path, custom_text, color_bgr, cancel_event)
        context.user_data.pop("cancel_event", None)

        if not success:
            await status_msg.edit_text("❌ Не удалось обработать видео или задача была отменена.")
            return

        await status_msg.edit_text("✅ Готово, отправляю результат...")
        # Prefer send as video_note if input was video_note, fallback to normal video
        media_kind = context.user_data.get("media_kind", "video")
        try:
            if media_kind == "video_note":
                # Try to send as video_note
                await msg.reply_video_note(video=InputFile(output_path))
            else:
                await msg.reply_video(video=InputFile(output_path))
        except Exception as e:
            logger.warning("Send as preferred media failed: %s — falling back to reply_video", e)
            try:
                await msg.reply_video(video=InputFile(output_path))
            except Exception as e2:
                logger.exception("Failed to send processed video: %s", e2)
                await msg.reply_text("Готово, но не удалось отправить файл — см. логи сервера.")
        try:
            await status_msg.delete()
        except Exception:
            pass
    finally:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass


# ---------- Telegram UI: color & text selection ----------


def color_keyboard(selected_key=None):
    keys = list(COLORS.keys())
    kb = []
    row = []
    for i, k in enumerate(keys):
        label = COLORS[k]["name"]
        if k == selected_key:
            label += " ✅"
        row.append(InlineKeyboardButton(label, callback_data=f"color|{k}"))
        if (i + 1) % 3 == 0:
            kb.append(row)
            row = []
    if row:
        kb.append(row)
    kb.append([InlineKeyboardButton("✏️ Задать текст", callback_data="set_text"),
               InlineKeyboardButton("▶️ Готово — отправить видео", callback_data="ready")])
    kb.append([InlineKeyboardButton("⛔ Cancel", callback_data="btn_cancel")])
    return InlineKeyboardMarkup(kb)


START_TEXT = "Привет! Нажми на цвет рамки и (опционально) задай текст для отображения. Затем отправь видео."

HELP_TEXT = (
    "Как пользоваться:\n"
    "1) Нажми Start -> выбери цвет и/или нажми «Задать текст» и пришли сообщение с текстом.\n"
    "2) Отправь видео (файл, видеосообщение или документ с видео). Длина до 20s (автоматически обрежу).\n"
    "3) Если нужно остановить обработку — нажми /cancel или кнопку Cancel.\n"
)


# Handlers


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.setdefault("color_key", "purple")
    context.user_data.setdefault("custom_text", None)
    await update.effective_message.reply_text(START_TEXT, reply_markup=color_keyboard(context.user_data["color_key"]))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(HELP_TEXT)
    else:
        await update.effective_message.reply_text(HELP_TEXT)


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await update.callback_query.answer()
        user_msg = update.callback_query.message
    else:
        user_msg = update.effective_message
    ev = context.user_data.get("cancel_event")
    if ev and isinstance(ev, threading.Event):
        ev.set()
        await user_msg.reply_text("⛔ Обработка отменена.")
    else:
        await user_msg.reply_text("Нет активной обработки.")


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("color|"):
        _, key = data.split("|", 1)
        if key in COLORS:
            context.user_data["color_key"] = key
            await query.message.edit_reply_markup(reply_markup=color_keyboard(key))
            await query.message.reply_text(f"Выбран цвет: {COLORS[key]['name']}")
        else:
            await query.message.reply_text("Неизвестный цвет.")
    elif data == "set_text":
        context.user_data["awaiting_text"] = True
        await query.message.reply_text("Пришли мне текст, который нужно отобразить в рамке (или отправь /cancel_text чтобы отменить).")
    elif data == "ready":
        await query.message.reply_text("Ок. Теперь отправь видео (файл, видеосообщение или документ с видео).")
    elif data == "btn_cancel":
        ev = context.user_data.get("cancel_event")
        if ev and isinstance(ev, threading.Event):
            ev.set()
            await query.message.reply_text("Отмена запрошена.")
        else:
            await query.message.reply_text("Нет активной задачи для отмены.")
    else:
        await query.message.reply_text("Неизвестная команда.")


async def cancel_text_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("awaiting_text", None)
    await update.effective_message.reply_text("Ввод текста отменён.")


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_text"):
        text = update.effective_message.text.strip()
        context.user_data["custom_text"] = text
        context.user_data.pop("awaiting_text", None)
        await update.effective_message.reply_text(f"Текст сохранён: {text}", reply_markup=color_keyboard(context.user_data.get("color_key")))
    else:
        await update.effective_message.reply_text("Я ожидаю видео. Нажми /help для инструкции.")


# Media handler


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    logger.info("Received media from user=%s video=%s video_note=%s document=%s", update.effective_user.id, bool(msg.video), bool(msg.video_note), bool(msg.document))
    file_obj = None
    media_kind = None
    if msg.video:
        file_obj = await msg.video.get_file()
        media_kind = "video"
    elif msg.video_note:
        file_obj = await msg.video_note.get_file()
        media_kind = "video_note"
    elif msg.document and (msg.document.mime_type or "").startswith("video"):
        file_obj = await msg.document.get_file()
        media_kind = "document"
    else:
        await msg.reply_text("Пожалуйста, отправь видео (файл, видеосообщение или документ с видео).")
        return

    tmpf = tempfile.NamedTemporaryFile(suffix=".input", delete=False)
    tmpf.close()
    try:
        await msg.reply_text("⬇️ Скачиваю видео...")
        await file_obj.download_to_drive(tmpf.name)
    except Exception as e:
        logger.exception("Download failed: %s", e)
        await msg.reply_text("Не удалось скачать файл. Попробуй ещё раз.")
        try:
            os.remove(tmpf.name)
        except Exception:
            pass
        return

    # store kind so run_and_send can decide how to send back
    context.user_data["media_kind"] = media_kind

    # choose color & text
    key = context.user_data.get("color_key", "purple")
    color_bgr = COLORS.get(key, COLORS["purple"])["bgr"]
    custom_text = context.user_data.get("custom_text")

    await run_and_send(update, context, tmpf.name, custom_text=custom_text, color_bgr=color_bgr)

    try:
        if os.path.exists(tmpf.name):
            os.remove(tmpf.name)
    except Exception:
        pass


# ---------- App building & run ----------


def build_app(token: str):
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("cancel_text", cancel_text_command))
    app.add_handler(CallbackQueryHandler(callback_query_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE | filters.Document.ALL, handle_media))
    return app


if __name__ == "__main__":
    application = build_app(BOT_TOKEN)
    try:
        if WEBHOOK_URL:
            url_path = BOT_TOKEN
            webhook_url = WEBHOOK_URL.rstrip("/") + "/" + url_path
            logger.info("Starting webhook mode on port %d webhook=%s", PORT, webhook_url)
            application.run_webhook(listen="0.0.0.0", port=PORT, url_path=url_path, webhook_url=webhook_url)
        else:
            logger.info("Starting polling mode")
            application.run_polling()
    except TGConflict:
        logger.warning("Conflict detected: trying delete_webhook() and fallback to polling")
        try:
            bot = application.bot
            asyncio.run(bot.delete_webhook(timeout=10))
            application.run_polling()
        except Exception as e:
            logger.exception("Failed to recover from Conflict: %s", e)
            raise
    except Exception as e:
        logger.exception("Unhandled startup error: %s", e)
        raise










