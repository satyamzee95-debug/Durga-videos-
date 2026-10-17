from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

# Put this file and durga.png in the same project folder.
# Optional: add devotional.mp3 for background music.
BASE = Path(__file__).resolve().parent
IMAGE_PATH = BASE / "durga.png"
AUDIO_PATH = BASE / "devotional.mp3"
OUTPUT_PATH = BASE / "maa_durga_drawing_reveal_1min.mp4"

W, H = 720, 1280
FPS = 24

# Updated timings for a 1-minute (60 seconds) video
TITLE_SECONDS = 5
SKETCH_SECONDS = 40
COLOR_SECONDS = 15
TOTAL_SECONDS = TITLE_SECONDS + SKETCH_SECONDS + COLOR_SECONDS


def fit_vertical(img):
    """Crop image to 9:16 without stretching."""
    h, w = img.shape[:2]
    target = W / H
    if w / h > target:
        nw = int(h * target)
        x = (w - nw) // 2
        img = img[:, x:x + nw]
    else:
        nh = int(w / target)
        y = (h - nh) // 2
        img = img[y:y + nh, :]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def load_font(size):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_title_frame(seconds):
    # Black title card similar to the supplied reference video.
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    title_font = load_font(32)
    subtitle_font = load_font(22)

    title = "Navratri Special"
    subtitle = "Maa Durga Drawing with Python"
    draw.text((W // 2, H // 2 - 30), title, font=title_font,
              fill=(255, 190, 45), anchor="mm")
    draw.text((W // 2, H // 2 + 20), subtitle, font=subtitle_font,
              fill=(255, 235, 185), anchor="mm")
    # A subtle progress line
    line_y = H // 2 + 75
    draw.rounded_rectangle((110, line_y, W - 110, line_y + 5),
                           radius=3, fill=(70, 55, 30))
    amount = min(1.0, seconds / TITLE_SECONDS)
    draw.rounded_rectangle((110, line_y, 110 + int((W - 220) * amount),
                            line_y + 5), radius=3, fill=(255, 180, 35))
    return np.array(pil)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Can't find {IMAGE_PATH.name}. Put your Maa Durga picture "
            "in this folder and name it durga.png."
        )

    print("Loading picture...")
    pil = Image.open(IMAGE_PATH).convert("RGB")
    pil = ImageEnhance.Color(pil).enhance(1.08)
    pil = ImageEnhance.Contrast(pil).enhance(1.05)
    color = fit_vertical(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Create a clean black line drawing on white paper.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 115)
    # Slightly thicken the illustration's lines.
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # A small warm paper tint like a drawing page.
    paper = np.full((H, W, 3), (244, 242, 229), dtype=np.uint8)
    paper[edges > 0] = (30, 30, 30)
    sketch = paper

    # Reveal linework in a gently wavy top-to-bottom sweep.
    yy, xx = np.mgrid[0:H, 0:W]
    sweep_map = yy + 18 * np.sin(xx / 48.0) + 8 * np.sin(xx / 17.0)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create MP4. Try: py -m pip install opencv-python")

    total_frames = int(TOTAL_SECONDS * FPS)
    sketch_frames = int(SKETCH_SECONDS * FPS)
    title_frames = int(TITLE_SECONDS * FPS)
    color_frames = int(COLOR_SECONDS * FPS)

    print("Rendering 1-minute drawing-reveal video. Please wait...")
    try:
        for idx in range(total_frames):
            if idx < title_frames:
                frame = make_title_frame(idx / FPS)
            elif idx < title_frames + sketch_frames:
                local = idx - title_frames
                progress = local / max(1, sketch_frames - 1)
                cutoff = int(progress * (H + 45))
                mask = sweep_map < cutoff
                # Unrevealed page stays clean parchment white.
                frame = np.full((H, W, 3), (244, 242, 229), dtype=np.uint8)
                frame[mask] = sketch[mask]
                # Slightly fade in the colored original near the end of sketch reveal.
                if progress > 0.86:
                    fade = min(1.0, (progress - 0.86) / 0.14)
                    frame = cv2.addWeighted(frame, 1.0 - 0.18 * fade,
                                            color, 0.18 * fade, 0)
            else:
                local = idx - title_frames - sketch_frames
                progress = local / max(1, color_frames - 1)
                # Fade from line art to the original colored devotional picture.
                frame = cv2.addWeighted(sketch, 1.0 - progress, color, progress, 0)

            # Slow gentle zoom throughout the illustration sections.
            if idx >= title_frames:
                t = (idx - title_frames) / FPS
                scale = 1.0 + 0.025 * min(1.0, t / (SKETCH_SECONDS + COLOR_SECONDS))
                sw, sh = int(W * scale), int(H * scale)
                enlarged = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
                x0, y0 = (sw - W) // 2, (sh - H) // 2
                frame = enlarged[y0:y0 + H, x0:x0 + W]

            writer.write(frame)
            if (idx + 1) % (FPS * 5) == 0:
                print(f"Rendered {idx + 1}/{total_frames} frames")
    finally:
        writer.release()

    # Optional soundtrack. If absent, the MP4 is still created without sound.
    if AUDIO_PATH.exists():
        try:
            from moviepy import VideoFileClip, AudioFileClip
            video = VideoFileClip(str(OUTPUT_PATH))
            audio = AudioFileClip(str(AUDIO_PATH))
            if audio.duration > TOTAL_SECONDS:
                audio = audio.subclipped(0, TOTAL_SECONDS)
            final = video.with_audio(audio)
            temp = BASE / "maa_durga_drawing_reveal_temp.mp4"
            final.write_videofile(
                str(temp), fps=FPS, codec="libx264", audio_codec="aac",
                preset="ultrafast", logger=None
            )
            final.close()
            audio.close()
            video.close()
            temp.replace(OUTPUT_PATH)
            print("Background music added.")
        except Exception as exc:
            print("Video created, but audio could not be added:", exc)

    print(f"\nDone! Saved 1-minute video to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()