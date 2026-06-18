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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
INDEX_TIP = 8


def is_l_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    index_up = lm[8].y < lm[6].y < lm[5].y
    middle_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_down and ring_down and pinky_down


def apply_rect_effect(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    result = frame.copy()
    region = frame[y1:y2, x1:x2].copy()

    inverted = cv2.bitwise_not(region)
    dark = (region * 0.12).astype(np.uint8)
    purple = inverted.copy()
    purple[:, :, 1] = (purple[:, :, 1] * 0.1).astype(np.uint8)
    blended = cv2.addWeighted(dark, 0.4, purple, 0.75, 0)
    glow = cv2.GaussianBlur(purple, (15, 15), 0)
    blended = cv2.addWeighted(blended, 1.0, glow, 0.35, 0)
    result[y1:y2, x1:x2] = blended

    border_color = (220, 0, 255)
    cv2.rectangle(result, (x1, y1), (x2, y2), border_color, 3)
    overlay = result.copy()
    cv2.rectangle(overlay, (x1-4, y1-4), (x2+4, y2+4), border_color, 8)
    result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
    return result


def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = input_path + "_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    prev_rect = None
    smooth = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        index_tips = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                if is_l_gesture(hand_lm):
                    tip = hand_lm.landmark[INDEX_TIP]
                    index_tips.append((int(tip.x * w), int(tip.y * h)))

        if len(index_tips) == 2:
            p1, p2 = index_tips
            rx1 = min(p1[0], p2[0])
            ry1 = min(p1[1], p2[1])
            rx2 = max(p1[0], p2[0])
            ry2 = max(p1[1], p2[1])

            if prev_rect is None:
                prev_rect = (rx1, ry1, rx2, ry2)
            else:
                rx1 = int(prev_rect[0] * (1-smooth) + rx1 * smooth)
                ry1 = int(prev_rect[1] * (1-smooth) + ry1 * smooth)
                rx2 = int(prev_rect[2] * (1-smooth) + rx2 * smooth)
                ry2 = int(prev_rect[3] * (1-smooth) + ry2 * smooth)
                prev_rect = (rx1, ry1, rx2, ry2)

            frame = apply_rect_effect(frame, rx1, ry1, rx2, ry2)
        else:
            prev_rect = None

        out.write(frame)

    cap.release()
    out.release()
    hands.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        cmd2 = ["ffmpeg", "-y", "-i", tmp_video, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
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
        "👋 Отправь кружок с L-жестом двумя руками — получишь пурпурный эффект внутри рамки 🟣"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
