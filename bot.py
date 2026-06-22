import os
import logging
import asyncio
import subprocess
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler,
    filters, ContextTypes
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands

INDEX_TIP = 8
THUMB_TIP = 4

COLORS = {
    "purple": {"name": "🟣 Пурпурный", "bgr": (255, 0, 200)},
    "blue":   {"name": "🔵 Синий",     "bgr": (255, 100, 0)},
    "green":  {"name": "🟢 Зелёный",   "bgr": (60, 255, 60)},
    "red":    {"name": "🔴 Красный",   "bgr": (40, 30, 230)},
    "yellow": {"name": "🟡 Жёлтый",    "bgr": (40, 230, 230)},
    "white":  {"name": "⚪ Белый",     "bgr": (240, 240, 240)},
}

# Путь к юникод-шрифту (DejaVu идёт в комплекте с большинством Linux-систем/Docker-образов)
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


def _finger_extended(lm, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
    """
    Проверяет вытянут ли палец, сравнивая расстояния от запястья,
    а не только Y-координаты — работает при любом повороте руки/камеры
    (важно для круглых кружков с fisheye-искажением).
    """
    wrist = np.array([lm[wrist_idx].x, lm[wrist_idx].y])
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y])
    mcp = np.array([lm[mcp_idx].x, lm[mcp_idx].y])

    dist_tip = np.linalg.norm(tip - wrist)
    dist_pip = np.linalg.norm(pip - wrist)
    dist_mcp = np.linalg.norm(mcp - wrist)

    # Палец вытянут, если кончик заметно дальше от запястья, чем сустав
    return dist_tip > dist_pip * 1.05 and dist_tip > dist_mcp * 1.15


def _finger_curled(lm, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
    """Проверяет согнут ли палец (кончик не дальше от запястья, чем средний сустав)"""
    wrist = np.array([lm[wrist_idx].x, lm[wrist_idx].y])
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y])

    dist_tip = np.linalg.norm(tip - wrist)
    dist_pip = np.linalg.norm(pip - wrist)

    return dist_tip < dist_pip * 1.15


def get_hand_points(hand_landmarks, w, h):
    lm = hand_landmarks.landmark

    # Указательный должен быть явно вытянут — это главный признак жеста
    index_extended = _finger_extended(lm, 8, 6, 5)
    if not index_extended:
        return None

    # Средний/безымянный/мизинец должны быть заметно короче указательного (согнуты).
    # Требуем минимум 2 из 3 — иначе открытая ладонь тоже проходила бы проверку.
    wrist = np.array([lm[0].x, lm[0].y])
    index_tip = np.array([lm[8].x, lm[8].y])
    dist_index = np.linalg.norm(index_tip - wrist)

    other_tips = [12, 16, 20]
    curled_count = 0
    for tip_idx in other_tips:
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
        dist_tip = np.linalg.norm(tip - wrist)
        if dist_tip < dist_index * 0.75:  # заметно короче, не просто чуть-чуть
            curled_count += 1

    if curled_count < 2:
        return None

    # Большой палец должен быть отведён в сторону от указательного (не прижат к ладони)
    thumb_tip = np.array([lm[THUMB_TIP].x, lm[THUMB_TIP].y])
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    if thumb_index_dist < dist_index * 0.3:  # слишком близко к указательному — не похоже на L
        return None

    ix = int(lm[INDEX_TIP].x * w)
    iy = int(lm[INDEX_TIP].y * h)
    tx = int(lm[THUMB_TIP].x * w)
    ty = int(lm[THUMB_TIP].y * h)
    return (ix, iy), (tx, ty)


def preprocess_for_detection(frame):
    """
    Улучшает кадр для более точного распознавания рук:
    - CLAHE для контраста (помогает в плохом освещении)
    - Лёгкая резкость
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Лёгкая нерезкая маска для повышения детализации краёв пальцев
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def detect_hands(hands_detector, frame, w, h):
    """
    Быстрая детекция рук с L-жестом на одном масштабе.
    Если кадр большой — даунскейлим перед детекцией (MediaPipe всё равно работает
    на низком внутреннем разрешении, гонять полный кадр — трата времени).
    Возвращает список (index_tip, thumb_tip) в координатах ОРИГИНАЛЬНОГО кадра.
    """
    MAX_SIDE = 640  # выше этого — даунскейлим для скорости; 640 достаточно и для кружков, и для видео
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
        logger.warning(f"MediaPipe упал на кадре, пропускаю: {e}")
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


def dedupe_hands(hand_points, dist_threshold=40):
    """Убирает дубликаты одной и той же руки, найденной на разных масштабах"""
    unique = []
    for idx_pt, thm_pt in hand_points:
        is_dup = False
        for u_idx, u_thm in unique:
            if (abs(idx_pt[0] - u_idx[0]) < dist_threshold and
                    abs(idx_pt[1] - u_idx[1]) < dist_threshold):
                is_dup = True
                break
        if not is_dup:
            unique.append((idx_pt, thm_pt))
    return unique


def pick_best_pair(hand_points, prev_center=None):
    """
    Если найдено больше 2 рук с L-жестом (несколько людей в кадре),
    выбирает пару, которая:
    1) Образует валидный (не вырожденный) четырёхугольник
    2) Если есть прошлая позиция — ближайшую к ней (стабильность между кадрами)
    3) Иначе — пару с наибольшей площадью получившейся фигуры
    """
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
            if area < 150:  # слишком маленький/вырожденный — пропускаем
                continue
            center = quad.mean(axis=0)
            candidates.append((area, center, a, b))
        except Exception:
            continue

    if not candidates:
        return None

    if prev_center is not None:
        # Выбираем пару с центром ближе к прошлому положению
        candidates.sort(key=lambda c: np.linalg.norm(c[1] - prev_center))
    else:
        # Иначе берём самую большую фигуру
        candidates.sort(key=lambda c: -c[0])

    _, _, a, b = candidates[0]
    return a, b


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


def draw_unicode_glow_text(img_bgr, text, center, box_width, color_bgr):
    """
    Рисует текст любого языка (кириллица, эмодзи, иероглифы и т.д.) через PIL
    с glow-эффектом, возвращает BGR numpy массив.
    """
    if not FONT_PATH:
        return img_bgr  # нет шрифта — пропускаем текст

    h, w = img_bgr.shape[:2]
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR -> RGB

    # Подбираем размер шрифта под ширину фигуры
    font_size = max(14, min(60, int(box_width / (len(text) * 0.62 + 1))))
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Слой для glow (RGBA)
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

    img_bgr[:] = (img_bgr.astype(np.float32) * (1 - glow_alpha) +
                  glow_bgr.astype(np.float32) * glow_alpha).astype(np.uint8)

    # Чёткий текст поверх
    sharp_np = cv2.cvtColor(np.array(glow_img), cv2.COLOR_RGBA2BGRA)
    sharp_bgr = sharp_np[:, :, :3]
    sharp_alpha = sharp_np[:, :, 3:4].astype(np.float32) / 255.0

    img_bgr[:] = (img_bgr.astype(np.float32) * (1 - sharp_alpha) +
                  sharp_bgr.astype(np.float32) * sharp_alpha).astype(np.uint8)

    return img_bgr


def apply_quad_effect(frame, pts, text=None, color_bgr=(255, 0, 200)):
    """
    Применяет эффект только в bounding box фигуры, а не на всём кадре —
    значительно быстрее и легче по памяти, особенно на больших видео.
    """
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
    cv2.polylines(glow_layer, [pts_local], True, color_bgr, 14)
    roi_result = cv2.addWeighted(roi_result, 0.72, glow_layer, 0.28, 0)
    cv2.polylines(roi_result, [pts_local], True, color_bgr, 2)

    frame[y1:y2, x1:x2] = roi_result
    return frame


def _process_video_inner(input_path: str, output_path: str, custom_text=None, color_bgr=(255, 0, 200)):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = input_path + "_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0,  # лёгкая модель — критично для скорости на слабом CPU; жест компенсирован мягкой проверкой
    )

    prev_pts = None
    prev_center = None
    smooth = 0.55          # выше = быстрее реагирует на резкие движения рук
    miss_count = 0
    MAX_MISS = 3          # короткое удержание (0.1 сек при 30fps) — не "зависает" после ухода рук
    MAX_JUMP = max(w, h) * 0.9  # резкие движения не должны отбрасываться как "выброс"

    frame_idx = 0
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    logger.info(f"Начинаю обработку: {w}x{h}, ~{total_frames_estimate} кадров, fps={fps:.2f}")

    while True:
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

                    if area >= 150:
                        if prev_center is None:
                            valid_update = True
                        else:
                            jump = np.linalg.norm(new_center - prev_center)
                            # При резком движении (большой скачок) доверяем новой позиции сразу,
                            # а не ждём — иначе фигура "не успевает" за быстрыми руками
                            valid_update = True

                    if valid_update:
                        if prev_pts is None:
                            prev_pts = quad.astype(np.float32)
                        else:
                            prev_pts = prev_pts * (1 - smooth) + quad.astype(np.float32) * smooth
                        prev_center = prev_pts.mean(axis=0)
                        frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                        miss_count = 0
                except Exception as e:
                    logger.warning(f"Кадр {frame_idx}: ошибка построения фигуры: {e}")
                    valid_update = False

            if not valid_update:
                miss_count += 1
                if prev_pts is not None and miss_count <= MAX_MISS:
                    frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                else:
                    prev_pts = None
                    prev_center = None
        except Exception as e:
            # Любая неожиданная ошибка на кадре — просто пишем кадр как есть, без эффекта,
            # и идём дальше. Никогда не обрываем обработку всего видео из-за одного кадра.
            logger.warning(f"Кадр {frame_idx}: непредвиденная ошибка, пишу без эффекта: {e}")

        out.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            logger.info(f"Прогресс: {frame_idx}/{total_frames_estimate} кадров")

    cap.release()
    out.release()
    hands.close()
    logger.info(f"Обработка кадров завершена: {frame_idx} кадров записано из ~{total_frames_estimate} ожидаемых")

    if total_frames_estimate > 0 and frame_idx < total_frames_estimate * 0.5:
        logger.error(
            f"Обработано подозрительно мало кадров ({frame_idx} из {total_frames_estimate}) — "
            f"вероятен сбой декодирования исходного видео или MediaPipe"
        )

    # Небольшая пауза, чтобы файловый дескриптор гарантированно сбросился на диск
    # перед тем как ffmpeg попытается его прочитать (актуально на медленных/нагруженных системах)
    import time
    time.sleep(0.3)

    # Проверяем что промежуточное видео реально записалось и не пустое
    if not os.path.exists(tmp_video) or os.path.getsize(tmp_video) < 1000:
        logger.error("Промежуточное видео не создалось или пустое")
        return False

    logger.info(f"Промежуточное видео создано: {os.path.getsize(tmp_video)} байт")

    cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", input_path,
           "-c:v", "libx264", "-c:a", "aac",
           "-map", "0:v:0", "-map", "1:a:0",
           "-shortest", "-pix_fmt", "yuv420p", output_path]
    try:
        r1 = subprocess.run(cmd, capture_output=True, timeout=300)
        if r1.returncode != 0:
            logger.error(f"ffmpeg (видео+аудио) код {r1.returncode}: {r1.stderr.decode(errors='ignore')[-800:]}")
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg (видео+аудио) превысил таймаут")

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        logger.info("Первый проход ffmpeg не дал результата, пробую без аудио")
        cmd2 = ["ffmpeg", "-y", "-i", tmp_video,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
        try:
            r2 = subprocess.run(cmd2, capture_output=True, timeout=300)
            if r2.returncode != 0:
                logger.error(f"ffmpeg (только видео) код {r2.returncode}: {r2.stderr.decode(errors='ignore')[-800:]}")
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg (только видео) превысил таймаут")

    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        logger.error("Финальный output.mp4 не создался или пустой после обоих проходов ffmpeg")
        return False

    logger.info(f"Финальное видео создано: {os.path.getsize(output_path)} байт")

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", output_path],
            capture_output=True, text=True, timeout=30
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            logger.error(f"ffprobe не подтвердил целостность файла: returncode={probe.returncode}, stderr={probe.stderr}")
            return False
    except Exception as e:
        logger.error(f"Ошибка проверки файла через ffprobe: {e}")
        return False

    logger.info("Видео успешно обработано и проверено")
    return True


def process_video(input_path: str, output_path: str, custom_text=None, color_bgr=(255, 0, 200)):
    """
    Безопасная обёртка: ловит АБСОЛЮТНО любое исключение (включая краши на конкретных кадрах,
    ошибки памяти, проблемы с кодеками) и логирует полный traceback, вместо того чтобы
    дать процессу тихо умереть без объяснений.
    """
    import traceback
    try:
        return _process_video_inner(input_path, output_path, custom_text, color_bgr)
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ошибка в process_video: {e}\n{traceback.format_exc()}")
        return False


async def run_and_send(update, context, input_path, custom_text=None, color_bgr=(255, 0, 200)):
    msg = update.effective_message
    status_msg = await msg.reply_text("🔄 Обрабатываю видео, это может занять некоторое время...")
    output_path = input_path + "_out.mp4"
    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, process_video, input_path, output_path, custom_text, color_bgr
        )

        if not success:
            await status_msg.edit_text(
                "❌ Не удалось обработать видео. Возможно, оно слишком длинное/тяжёлое — "
                "попробуй покороче (до 15-20 секунд) или с меньшим разрешением."
            )
            return

        media_kind = context.user_data.get("media_kind", "video_note")

        try:
            if media_kind == "video_note":
                vn = context.user_data.get("video_note_meta", {})
                with open(output_path, "rb") as f:
                    await context.bot.send_video_note(
                        chat_id=msg.chat_id,
                        video_note=f,
                        duration=vn.get("duration"),
                        length=vn.get("length"),
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60,
                    )
            else:
                with open(output_path, "rb") as f:
                    await context.bot.send_video(
                        chat_id=msg.chat_id,
                        video=f,
                        supports_streaming=True,
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60,
                    )
            await status_msg.delete()
        except Exception as e:
            logger.error(f"Ошибка отправки видео: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ Готово, но не получилось отправить: {e}")
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Ошибка обработки: {e}")
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.remove(p)


MAX_DURATION_SECONDS = 60  # ограничение на длину видео — длиннее будет обрабатываться слишком долго


async def handle_video_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    if msg.video_note.duration and msg.video_note.duration > MAX_DURATION_SECONDS:
        await msg.reply_text(
            f"⚠️ Видео слишком длинное ({msg.video_note.duration} сек). "
            f"Максимум {MAX_DURATION_SECONDS} секунд — иначе обработка займёт слишком много времени."
        )
        return

    file = await context.bot.get_file(msg.video_note.file_id)

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.mp4")
    await file.download_to_drive(input_path)

    context.user_data["pending_input_path"] = input_path
    context.user_data["media_kind"] = "video_note"
    context.user_data["awaiting_text"] = False
    context.user_data["video_note_meta"] = {
        "duration": msg.video_note.duration,
        "length": msg.video_note.length,
    }

    keyboard = build_color_keyboard()
    await msg.reply_text("🎨 Выбери цвет эффекта:", reply_markup=keyboard)


async def handle_regular_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    if msg.video.duration and msg.video.duration > MAX_DURATION_SECONDS:
        await msg.reply_text(
            f"⚠️ Видео слишком длинное ({msg.video.duration} сек). "
            f"Максимум {MAX_DURATION_SECONDS} секунд — иначе обработка займёт слишком много времени."
        )
        return

    file = await context.bot.get_file(msg.video.file_id)

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.mp4")
    await file.download_to_drive(input_path)

    context.user_data["pending_input_path"] = input_path
    context.user_data["media_kind"] = "video"
    context.user_data["awaiting_text"] = False

    keyboard = build_color_keyboard()
    await msg.reply_text("🎨 Выбери цвет эффекта:", reply_markup=keyboard)


def build_color_keyboard():
    buttons = []
    row = []
    for key, val in COLORS.items():
        row.append(InlineKeyboardButton(val["name"], callback_data=f"color_{key}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


async def handle_color_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    color_key = query.data.replace("color_", "")
    color_bgr = COLORS.get(color_key, COLORS["purple"])["bgr"]
    context.user_data["chosen_color"] = color_bgr

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✍️ Добавить текст", callback_data="add_text")],
        [InlineKeyboardButton("🚫 Без текста", callback_data="no_text")],
    ])
    await query.edit_message_text("Хочешь добавить текст внутрь рамки?", reply_markup=keyboard)


async def handle_text_choice_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    input_path = context.user_data.get("pending_input_path")
    color_bgr = context.user_data.get("chosen_color", (255, 0, 200))

    if not input_path or not os.path.exists(input_path):
        await query.edit_message_text("❌ Файл не найден, отправь кружок/видео заново")
        return

    if query.data == "no_text":
        context.user_data["awaiting_text"] = False
        await query.edit_message_text("🔄 Обрабатываю без текста...")
        await run_and_send(update, context, input_path, custom_text=None, color_bgr=color_bgr)

    elif query.data == "add_text":
        context.user_data["awaiting_text"] = True
        await query.edit_message_text("✏️ Напиши текст на любом языке — он появится в рамке:")


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Реагируем на обычный текст только если реально ждём текст для рамки —
    # иначе бот будет пытаться обработать любое случайное сообщение пользователя
    if not context.user_data.get("awaiting_text"):
        return

    text = update.message.text.strip()
    input_path = context.user_data.get("pending_input_path")
    color_bgr = context.user_data.get("chosen_color", (255, 0, 200))

    if not input_path or not os.path.exists(input_path):
        await update.message.reply_text("❌ Файл не найден, отправь кружок/видео заново")
        context.user_data["awaiting_text"] = False
        return

    if len(text) > 25:
        await update.message.reply_text("⚠️ Текст слишком длинный (максимум 25 символов), напиши короче:")
        return

    context.user_data["awaiting_text"] = False
    await run_and_send(update, context, input_path, custom_text=text, color_bgr=color_bgr)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input_path = context.user_data.get("pending_input_path")
    if input_path and os.path.exists(input_path):
        os.remove(input_path)
    context.user_data.clear()
    await update.message.reply_text("Отменено.")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Отправь кружок (видеосообщение) или обычное видео с L-жестом двумя руками!\n\n"
        "🤙 Указательный вверх + большой в сторону — обе руки, работает с любого расстояния.\n"
        "После этого выберешь цвет эффекта и можно добавить текст на любом языке.\n\n"
        "/cancel — отменить текущую операцию"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    app.add_handler(MessageHandler(filters.VIDEO, handle_regular_video))
    app.add_handler(CallbackQueryHandler(handle_color_choice, pattern="^color_"))
    app.add_handler(CallbackQueryHandler(handle_text_choice_buttons, pattern="^(add_text|no_text)$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))

    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()


