import os
import logging
import asyncio
import subprocess
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler,
    filters, ContextTypes, ConversationHandler
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands

INDEX_TIP = 8
THUMB_TIP = 4

WAITING_TEXT = 1
WAITING_COLOR = 2

# Цвета в формате BGR (OpenCV)
COLORS = {
    "purple": {"name": "🟣 Пурпурный", "bgr": (255, 0, 200)},
    "blue":   {"name": "🔵 Синий",     "bgr": (255, 100, 0)},
    "green":  {"name": "🟢 Зелёный",   "bgr": (60, 255, 60)},
    "red":    {"name": "🔴 Красный",   "bgr": (40, 30, 230)},
    "yellow": {"name": "🟡 Жёлтый",    "bgr": (40, 230, 230)},
    "white":  {"name": "⚪ Белый",     "bgr": (240, 240, 240)},
}


def get_hand_points(hand_landmarks, w, h):
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


def draw_glow_text(img, text, center, font_scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)

    overlay = np.zeros_like(img)
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness + 4, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)

    img[:] = cv2.addWeighted(img, 1.0, overlay, 0.8, 0)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def apply_quad_effect(frame, pts, text=None, color_bgr=(255, 0, 200)):
    H, W = frame.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    dark = (frame * 0.06).astype(np.uint8)

    # Тонируем инвертированное изображение в выбранный цвет
    inv = cv2.bitwise_not(frame)
    gray_inv = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    tinted = np.zeros_like(frame)
    for i in range(3):
        tinted[:, :, i] = (gray_inv.astype(np.float32) * (color_bgr[i] / 255.0)).astype(np.uint8)

    glow = cv2.GaussianBlur(tinted, (31, 31), 0)

    effect = cv2.addWeighted(dark, 0.15, tinted, 0.55, 0)
    effect = cv2.addWeighted(effect, 1.0, glow, 0.45, 0)

    if text:
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        quad_w = max(pts[:, 0]) - min(pts[:, 0])
        font_scale = max(0.5, min(1.6, quad_w / (len(text) * 28 + 1)))
        draw_glow_text(effect, text, (cx, cy), font_scale, color_bgr)

    result = (frame.astype(np.float32) * (1 - mask3) + effect.astype(np.float32) * mask3).astype(np.uint8)

    glow_layer = result.copy()
    cv2.polylines(glow_layer, [pts], True, color_bgr, 14)
    result = cv2.addWeighted(result, 0.72, glow_layer, 0.28, 0)
    cv2.polylines(result, [pts], True, color_bgr, 2)

    return result


def process_video(input_path: str, output_path: str, custom_text=None, color_bgr=(255, 0, 200)):
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
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                pts = get_hand_points(hand_lm, w, h)
                if pts is not None:
                    hand_points.append(pts)

        if len(hand_points) == 2:
            try:
                quad = make_quad(hand_points[0], hand_points[1])
                if prev_pts is None:
                    prev_pts = quad.astype(np.float32)
                else:
                    prev_pts = prev_pts * (1 - smooth) + quad.astype(np.float32) * smooth
                frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
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


async def run_and_send(update, context, input_path, custom_text=None, color_bgr=(255, 0, 200)):
    msg = update.effective_message
    await msg.reply_text("🔄 Обрабатываю...")
    output_path = input_path + "_out.mp4"
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_video, input_path, output_path, custom_text, color_bgr)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            await msg.reply_text("❌ Не удалось обработать")
            return
        vn = context.user_data.get("video_note_meta", {})
        with open(output_path, "rb") as f:
            await context.bot.send_video_note(
                chat_id=msg.chat_id,
                video_note=f,
                duration=vn.get("duration"),
                length=vn.get("length"),
            )
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.remove(p)


async def handle_video_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    msg = update.message
    file = await context.bot.get_file(msg.video_note.file_id)

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.mp4")
    await file.download_to_drive(input_path)

    context.user_data["pending_input_path"] = input_path
    context.user_data["video_note_meta"] = {
        "duration": msg.video_note.duration,
        "length": msg.video_note.length,
    }

    keyboard = build_color_keyboard()
    await msg.reply_text("🎨 Выбери цвет эффекта:", reply_markup=keyboard)
    return WAITING_COLOR


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


async def handle_color_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
    return WAITING_TEXT


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    input_path = context.user_data.get("pending_input_path")
    color_bgr = context.user_data.get("chosen_color", (255, 0, 200))

    if not input_path or not os.path.exists(input_path):
        await query.edit_message_text("❌ Файл не найден, отправь кружок заново")
        return ConversationHandler.END

    if query.data == "no_text":
        await query.edit_message_text("🔄 Обрабатываю без текста...")
        await run_and_send(update, context, input_path, custom_text=None, color_bgr=color_bgr)
        return ConversationHandler.END

    elif query.data == "add_text":
        await query.edit_message_text("✏️ Напиши текст, который появится в рамке:")
        return WAITING_TEXT


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    input_path = context.user_data.get("pending_input_path")
    color_bgr = context.user_data.get("chosen_color", (255, 0, 200))

    if not input_path or not os.path.exists(input_path):
        await update.message.reply_text("❌ Файл не найден, отправь кружок заново")
        return ConversationHandler.END

    if len(text) > 20:
        await update.message.reply_text("⚠️ Текст слишком длинный (максимум 20 символов), напиши короче:")
        return WAITING_TEXT

    await run_and_send(update, context, input_path, custom_text=text, color_bgr=color_bgr)
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    input_path = context.user_data.get("pending_input_path")
    if input_path and os.path.exists(input_path):
        os.remove(input_path)
    await update.message.reply_text("Отменено.")
    return ConversationHandler.END


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Отправь кружок с L-жестом двумя руками!\n\n"
        "🤙 Указательный вверх + большой в сторону — обе руки.\n"
        "После этого выберешь цвет эффекта и можно добавить текст.\n\n"
        "/cancel — отменить текущую операцию"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.VIDEO_NOTE, handle_video_note)],
        states={
            WAITING_COLOR: [CallbackQueryHandler(handle_color_choice, pattern="^color_")],
            WAITING_TEXT: [
                CallbackQueryHandler(handle_button),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(conv_handler)

    logger.info("Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()



