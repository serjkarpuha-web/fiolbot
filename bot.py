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


def get_hand_points(hand_landmarks, w, h, label):
    """Возвращает (index_tip, thumb_tip) если L-жест"""
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


def make_quad(left_hand, right_hand):
    """
    Строим четырёхугольник правильно:
    Левая рука: большой палец = левый нижний угол, указательный = левый верхний
    Правая рука: большой палец = правый нижний угол, указательный = правый верхний
    Порядок: верх-лево, верх-право, низ-право, низ-лево (для корректного polylines)
    """
    l_idx, l_thm = left_hand   # левая: указательный, большой
    r_idx, r_thm = right_hand  # правая: указательный, большой

    # Определяем какая рука левая/правая по X-координате большого пальца
    # (большой палец левой руки правее чем указательный, правой — левее)
    # Проще: берём руку с меньшим X центра как левую
    l_cx = (l_idx[0] + l_thm[0]) // 2
    r_cx = (r_idx[0] + r_thm[0]) // 2

    if l_cx > r_cx:
        left_hand, right_hand = right_hand, left_hand
        l_idx, l_thm = left_hand
        r_idx, r_thm = right_hand

    # Четыре угла: верх-лево=левый указательный, верх-право=правый указательный
    #              низ-лево=левый большой, низ-право=правый большой
    tl = np.array(l_idx)
    tr = np.array(r_idx)
    br = np.array(r_thm)
    bl = np.array(l_thm)

    return np.array([tl, tr, br, bl], dtype=np.int32)


def apply_quad_effect(frame, pts):
    """
    Внутри четырёхугольника:
    - тёмный фон (почти чёрный)
    - пурпурный силуэт (инверт без зелёного)
    - тонкая пурпурная рамка с glow
    """
    H, W = frame.shape[:2]
    result = frame.copy()

    # Маска
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Тёмный фон
    dark = (frame * 0.06).astype(np.uint8)

    # Пурпурный инверт
    inv = cv2.bitwise_not(frame)
    purple = inv.copy()
    purple[:, :, 1] = 0   # убить зелёный
    purple[:, :, 2] = (purple[:, :, 2] * 0.7).astype(np.uint8)  # приглушить красный

    # Мягкий glow
    glow = cv2.GaussianBlur(purple, (31, 31), 0)

    # Финальный эффект = тёмный + пурпур + glow
    effect = cv2.addWeighted(dark, 0.15, purple, 0.55, 0)
    effect = cv2.addWeighted(effect, 1.0, glow, 0.45, 0)

    # Накладываем по маске
    result = (frame.astype(np.float32) * (1 - mask3) + effect.astype(np.float32) * mask3).astype(np.uint8)

    # Рамка с glow
    color = (255, 0, 200)
    glow_layer = result.copy()
    cv2.polylines(glow_layer, [pts], True, color, 14)
    result = cv2.addWeighted(result, 0.72, glow_layer, 0.28, 0)
    cv2.polylines(result, [pts], True, color, 2)

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

    prev_pts = None
    smooth = 0.45

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_points = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # 'Left' or 'Right'
                pts = get_hand_points(hand_lm, w, h, label)
                if pts is not None:
                    hand_points.append(pts)

        if len(hand_points) == 2:
            try:
                quad = make_quad(hand_points[0], hand_points[1])

                if prev_pts is None:
                    prev_pts = quad.astype(np.float32)
                else:
                    prev_pts = prev_pts * (1 - smooth) + quad.astype(np.float32) * smooth

                frame = apply_quad_effect(frame, prev_pts.astype(np.int32))
            except Exception:
                pass
        else:
            prev_pts = None

        out.write(frame)

    cap.release()
    out.release()
    hands.close()

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
        "🤙 Указательный вверх + большой в сторону — обе руки.\n"
        "Форма строится по 4 пальцам, можно наклонять и менять!"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
    
