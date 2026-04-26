#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

out = Path('assets/052-rq1-tfidf-confusion-matrix.png')
out.parent.mkdir(parents=True, exist_ok=True)
W, H = 1200, 800
img = Image.new('RGB', (W, H), color=(255, 255, 255))
d = ImageDraw.Draw(img)
try:
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 28)
except:
    fnt = ImageFont.load_default()

text = 'RQ1 — TF-IDF + Logistic Regression\nConfusion Matrix (placeholder)'
# center text
lines = text.split('\n')
y = 200
for line in lines:
    try:
        w, h = fnt.getsize(line)
    except AttributeError:
        # fallback for very new Pillow versions
        bbox = d.textbbox((0,0), line, font=fnt)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    d.text(((W - w) / 2, y), line, font=fnt, fill=(0, 0, 0))
    y += h + 10

# draw a faux matrix grid
grid_x0, grid_y0 = 150, 350
cell = 38
for i in range(15):
    for j in range(15):
        x0 = grid_x0 + j * cell
        y0 = grid_y0 + i * cell
        d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(180,180,180))
# label axes
try:
    small = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
except:
    small = ImageFont.load_default()

d.text((grid_x0, grid_y0 - 30), 'Confusion matrix (rows=true, cols=pred)', font=small, fill=(0,0,0))
img.save(out)
print('Saved placeholder:', out)
