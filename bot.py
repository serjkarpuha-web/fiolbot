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


def get_hand_points(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    # Более надёжная проверка L-жеста с запасом (используем длины фаланг, не только y)
    index_up = (lm[8].y < lm[6].y) and (lm[6].y < lm[5].y)
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


def detect_hands_multiscale(hands_detector, frame, w, h):
    """
    Пытается найти руки с L-жестом на нескольких масштабах кадра,
    чтобы ловить как крупные (близко), так и мелкие (далеко) руки.
    Возвращает список (index_tip, thumb_tip) точек уже в координатах оригинального кадра.
    """
    scales = [1.0, 1.5, 0.7]
    found = []

    for scale in scales:
        if len(found) >= 2:
            break

        if scale == 1.0:
            scaled = frame
            sw, sh = w, h
        else:
            sw, sh = int(w * scale), int(h * scale)
            scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                pts = get_hand_points(hand_lm, sw, sh)
                if pts is not None:
                    idx_pt, thm_pt = pts
                    # Переводим обратно в координаты оригинала
                    idx_orig = (int(idx_pt[0] / scale), int(idx_pt[1] / scale))
                    thm_orig = (int(thm_pt[0] / scale), int(thm_pt[1] / scale))
                    found.append((idx_orig, thm_orig))
                    if len(found) >= 2:
                        break

    return found


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
    glow_blur = cv2.GaussianBlur(glow_np, (15, 15), 0)
    glow_bgr = glow_blur[:, :, :3]
    glow_alpha = (glow_blur[:, :, 3:4].astype(np.float32) / 255.0) * 0.8

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
    H, W = frame.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    dark = (frame * 0.06).astype(np.uint8)

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
        quad_w = int(max(pts[:, 0]) - min(pts[:, 0]))
        effect = draw_unicode_glow_text(effect, text, (cx, cy), quad_w, color_bgr)

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

    # Максимальная точность: model_complexity=1 (полная модель), низкий порог уверенности
    # чтобы ловить руки на разном расстоянии, в разном освещении
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
        model_complexity=1,
    )

    prev_pts = None
    smooth = 0.4
    miss_count = 0
    MAX_MISS = 3  # сколько кадров держим прошлую фигуру если руки временно потерялись

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Улучшаем кадр перед детекцией (но эффект применяем на оригинальный frame)
        enhanced = preprocess_for_detection(frame)
        hand_points = detect_hands_multiscale(hands, enhanced, w, h)

        if len(hand_points) == 2:
            try:
                quad = make_quad(hand_points[0], hand_points[1])
                if prev_pts is None:
                    prev_pts = quad.astype(np.float32)
                else:
                    prev_pts = prev_pts * (1 - smooth) + quad.astype(np.float32) * smooth
                frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
                miss_count = 0
            except Exception:
                pass
        else:
            miss_count += 1
            if prev_pts is not None and miss_count <= MAX_MISS:
                # Держим последнюю известную фигуру короткое время (борьба с дрожанием детекции)
                frame = apply_quad_effect(frame, prev_pts.astype(np.int32), custom_text, color_bgr)
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

        media_kind = context.user_data.get("media_kind", "video_note")

        if media_kind == "video_note":
            vn = context.user_data.get("video_note_meta", {})
            with open(output_path, "rb") as f:
                await context.bot.send_video_note(
                    chat_id=msg.chat_id,
                    video_note=f,
                    duration=vn.get("duration"),
                    length=vn.get("length"),
                )
        else:
            with open(output_path, "rb") as f:
                await context.bot.send_video(
                    chat_id=msg.chat_id,
                    video=f,
                    supports_streaming=True,
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
    context.user_data["media_kind"] = "video_note"
    context.user_data["video_note_meta"] = {
        "duration": msg.video_note.duration,
        "length": msg.video_note.length,
    }

    keyboard = build_color_keyboard()
    await msg.reply_text("🎨 Выбери цвет эффекта:", reply_markup=keyboard)
    return WAITING_COLOR


async def handle_regular_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    msg = update.message
    file = await context.bot.get_file(msg.video.file_id)

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, "input.mp4")
    await file.download_to_drive(input_path)

    context.user_data["pending_input_path"] = input_path
    context.user_data["media_kind"] = "video"

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
        await query.edit_message_text("✏️ Напиши текст на любом языке — он появится в рамке:")
        return WAITING_TEXT


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    input_path = context.user_data.get("pending_input_path")
    color_bgr = context.user_data.get("chosen_color", (255, 0, 200))

    if not input_path or not os.path.exists(input_path):
        await update.message.reply_text("❌ Файл не найден, отправь кружок заново")
        return ConversationHandler.END

    if len(text) > 25:
        await update.message.reply_text("⚠️ Текст слишком длинный (максимум 25 символов), напиши короче:")
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
        "👋 Отправь кружок (видеосообщение) или обычное видео с L-жестом двумя руками!\n\n"
        "🤙 Указательный вверх + большой в сторону — обе руки, работает с любого расстояния.\n"
        "После этого выберешь цвет эффекта и можно добавить текст на любом языке.\n\n"
        "/cancel — отменить текущую операцию"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.VIDEO_NOTE, handle_video_note),
            MessageHandler(filters.VIDEO, handle_regular_video),
        ],
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


