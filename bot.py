import os
import logging
import asyncio
import subprocess
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands

INDEX_TIP = 8
THUMB_TIP = 4


def get_hand_points(hand_landmarks, w, h):
    """Возвращает (index_tip, thumb_tip) если L-жест, иначе None"""
    lm = hand_landmarks.landmark
    index_up = lm[8].y < lm[6].y < lm[5].y
    middle_down = lm[12].y > lm[10].y
    ring_down   = lm[16].y > lm[14].y
    pinky_down  = lm[20].y > lm[18].y
    if not (index_up and middle_down and ring_down and pinky_down):
        return None
    ix = int(lm[INDEX_TIP].x * w)
    iy = int(lm[INDEX_TIP].y * h)
    tx = int(lm[THUMB_TIP].x * w)
    ty = int(lm[THUMB_TIP].y * h)
    return (ix, iy), (tx, ty)


def apply_quad_effect(frame, pts):
    """
    pts: 4 точки четырёхугольника (numpy array shape (4,2))
    Внутри: тёмный фон + пурпурный силуэт (как тепловизор)
    Рамка: тонкая пурпурная с glow
    """
    H, W = frame.shape[:2]
    result = frame.copy()

    # Маска четырёхугольника
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Регион внутри
    region = frame.copy()

    # 1. Тёмный фон — почти чёрный
    dark = (region * 0.05).astype(np.uint8)

    # 2. Пурпурный инверт — убираем зелёный канал
    inverted = cv2.bitwise_not(region)
    purple = inverted.copy()
    purple[:, :, 1] = (purple[:, :, 1] * 0.0).astype(np.uint8)   # убить зелёный полностью
    purple[:, :, 2] = (purple[:, :, 2] * 0.85).astype(np.uint8)  # немного убрать красный

    # 3. Размытый glow
    glow = cv2.GaussianBlur(purple, (25, 25), 0)

    # 4. Смешиваем: тёмный + пурпур + glow
    blended = cv2.addWeighted(dark, 0.2, purple, 0.6, 0)
    blended = cv2.addWeighted(blended, 1.0, glow, 0.5, 0)

    # 5. Применяем только внутри маски
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_f = mask3.astype(np.float32) / 255.0
    result = (frame.astype(np.float32) * (1 - mask_f) + blended.astype(np.float32) * mask_f).astype(np.uint8)

    # 6. Рамка — тонкая пурпурная с glow
    border_color = (255, 0, 200)

    # Glow рамки (широкая полупрозрачная)
    glow_layer = result.copy()
    cv2.polylines(glow_layer, [pts], True, border_color, 12)
    result = cv2.addWeighted(result, 0.75, glow_layer, 0.25, 0)

    # Основная рамка (тонкая чёткая)
    cv2.polylines(result, [pts], True, border_color, 2)

    return result


def order_points(pts_list):
    """
    pts_list: список из 4 точек [(x,y), ...]
    Упорядочивает по часовой: верх-лево, верх-право, низ-право, низ-лево
    """
    pts = np.array(pts_list, dtype=np.int32)
    # Сортируем по сумме координат
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def process_video(input_path: str, output_path: str):
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
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    prev_pts = None
    smooth = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_points = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                pts = get_hand_points(hand_lm, w, h)
                if pts is not None:
                    hand_points.append(pts)

        if len(hand_points) == 2:
            # 4 точки: index и thumb каждой руки
            p0_idx, p0_thm = hand_points[0]
            p1_idx, p1_thm = hand_points[1]
            raw_pts = [p0_idx, p0_thm, p1_idx, p1_thm]

            ordered = order_points(raw_pts)

            # Сглаживание
            if prev_pts is None:
                prev_pts = ordered.astype(np.float32)
            else:
                prev_pts = prev_pts * (1 - smooth) + ordered.astype(np.float32) * smooth

            quad = prev_pts.astype(np.int32)
            frame = apply_quad_effect(frame, quad)
        else:
            prev_pts = None

        out.write(frame)

    cap.release()
    out.release()
    hands.close()

    # Склеиваем с аудио
    cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", input_path,
           "-c:v", "libx264", "-c:a", "aac",
           "-map", "0:v:0", "-map", "1:a:0",
           "-shortest", "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, capture_output=True)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        cmd2 = ["ffmpeg", "-y", "-i", tmp_video,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
        subprocess.run(cmd2, capture_output=True)

    if os.path.exists(tmp_video):
        os.remove(tmp_video)


async def handle_video_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    await msg.reply_text("🔄 Обрабатываю...")
    try:
        file = await context.bot.get_file(msg.video_note.file_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            await file.download_to_drive(input_path)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, process_video, input_path, output_path)
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                await msg.reply_text("❌ Не удалось обработать")
                return
            with open(output_path, "rb") as f:
                await context.bot.send_video_note(
                    chat_id=msg.chat_id,
                    video_note=f,
                    duration=msg.video_note.duration,
                    length=msg.video_note.length,
                )
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        await msg.reply_text(f"❌ Ошибка: {e}")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Отправь кружок с L-жестом двумя руками!\n\n"
        "🤙 Жест: указательный вверх + большой в сторону (обе руки)\n"
        "Форма строится по 4 пальцам — можешь наклонять и менять форму!"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
