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

# Индексы ландмарков
INDEX_TIP = 8   # кончик указательного
THUMB_TIP = 4   # кончик большого


def get_hand_points(hand_landmarks, w, h):
    """Возвращает (index_tip, thumb_tip) в пикселях если L-жест, иначе None"""
    lm = hand_landmarks.landmark

    # Указательный поднят
    index_up = lm[8].y < lm[6].y < lm[5].y
    # Остальные (кроме большого) согнуты
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


def apply_effect(frame, x1, y1, x2, y2):
    """Внутри прямоугольника: тёмный фон + пурпурный инвертированный эффект + пурпурная рамка с glow"""
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    result = frame.copy()
    region = frame[y1:y2, x1:x2].copy()

    # Инвертируем
    inverted = cv2.bitwise_not(region)

    # Убираем зелёный канал → пурпур
    purple = inverted.copy()
    purple[:, :, 1] = (purple[:, :, 1] * 0.05).astype(np.uint8)

    # Тёмный фон
    dark = (region * 0.08).astype(np.uint8)

    # Смешиваем
    blended = cv2.addWeighted(dark, 0.3, purple, 0.85, 0)

    # Glow поверх
    glow = cv2.GaussianBlur(purple, (21, 21), 0)
    blended = cv2.addWeighted(blended, 1.0, glow, 0.4, 0)

    result[y1:y2, x1:x2] = blended

    # Пурпурная рамка
    color = (255, 0, 220)
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

    # Glow рамки
    glow_layer = result.copy()
    cv2.rectangle(glow_layer, (x1-5, y1-5), (x2+5, y2+5), color, 10)
    cv2.rectangle(glow_layer, (x1-10, y1-10), (x2+10, y2+10), color, 6)
    result = cv2.addWeighted(result, 0.7, glow_layer, 0.3, 0)

    return result


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

    prev_rect = None
    smooth = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Собираем точки обеих рук
        all_points = []  # список (ix,iy,tx,ty) для каждой руки с L-жестом

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                pts = get_hand_points(hand_lm, w, h)
                if pts is not None:
                    all_points.append(pts)

        if len(all_points) == 2:
            # Берём все 4 точки: 2 указательных + 2 больших
            pts_flat = [all_points[0][0], all_points[0][1],
                        all_points[1][0], all_points[1][1]]
            xs = [p[0] for p in pts_flat]
            ys = [p[1] for p in pts_flat]

            rx1, ry1 = min(xs), min(ys)
            rx2, ry2 = max(xs), max(ys)

            # Сглаживание
            if prev_rect is None:
                prev_rect = (rx1, ry1, rx2, ry2)
            else:
                rx1 = int(prev_rect[0] * (1-smooth) + rx1 * smooth)
                ry1 = int(prev_rect[1] * (1-smooth) + ry1 * smooth)
                rx2 = int(prev_rect[2] * (1-smooth) + rx2 * smooth)
                ry2 = int(prev_rect[3] * (1-smooth) + ry2 * smooth)
                prev_rect = (rx1, ry1, rx2, ry2)

            frame = apply_effect(frame, rx1, ry1, rx2, ry2)
        else:
            prev_rect = None

        out.write(frame)

    cap.release()
    out.release()
    hands.close()

    # Склеиваем с аудио
    cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", input_path,
           "-c:v", "libx264", "-c:a", "aac",
           "-map", "0:v:0", "-map", "1:a:0",
           "-shortest", "-pix_fmt", "yuv420p", output_path]
    r = subprocess.run(cmd, capture_output=True)

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
        "👋 Отправь кружок!\n\n"
        "🤙 Держи обе руки L-жестом (указательный вверх + большой в сторону).\n"
        "Прямоугольник строится между всеми 4 пальцами."
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
