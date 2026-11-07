# Draws the deterministic test image used by the vision checks (tests/vision_test.png, 320x240: the height
# is resized to 256, so the bicubic resampler is exercised).
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (320, 240), (235, 240, 250))
d = ImageDraw.Draw(img)
d.rectangle([20, 30, 120, 130], fill=(220, 30, 30))             # red square
d.ellipse([190, 40, 290, 140], fill=(30, 90, 220))             # blue circle
d.polygon([(70, 220), (130, 150), (190, 220)], fill=(40, 170, 60))  # green triangle
try:
    font = ImageFont.truetype("arial.ttf", 28)
except OSError:
    font = ImageFont.load_default()
d.text((200, 170), "QWEN 42", fill=(10, 10, 10), font=font)
img.save("tests/vision_test.png")
print("wrote tests/vision_test.png")
