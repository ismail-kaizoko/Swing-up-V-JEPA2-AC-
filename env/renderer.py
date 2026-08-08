"""
Rasterizes a pendulum angle into a small square grayscale image. Uses PIL's
ImageDraw rather than matplotlib — roughly 50-100x faster per frame, which
matters once you're generating tens of thousands of frames for pretraining.
"""


import numpy as np
from PIL import Image, ImageDraw


class PendulumRenderer:
    def __init__(self, render_size: int = 64, final_size = 32,  rod_length_px: int =0.45,
                 line_width: int = 0.01, bob_radius: int = 0.05):
        self.render_size = render_size
        self.img_size = final_size
        self.center = tuple(render_size*np.array((0.5 , 0.3)))
        self.rod_length_px = int(render_size*rod_length_px)
        self.line_width = int(render_size*line_width)
        self.bob_radius = int(render_size*bob_radius)

    def render(self, theta: float) -> np.ndarray:
        img = Image.new("L", (self.render_size, self.render_size), color=0)  # "L" = 8-bit grayscale
        draw = ImageDraw.Draw(img)
        cx, cy = tuple(int(x) for x in self.center)

        # theta=0 is upright: in image coordinates y grows *downward*, so
        # the bob offset from the pivot is (sin(theta), -cos(theta)) —
        # at theta=0 that's (0, -1): straight up on screen, as intended.
        bx = cx + self.rod_length_px * np.sin(theta)
        by = cy - self.rod_length_px * np.cos(theta)

        draw.line([(cx, cy), (bx, by)], fill=255, width=self.line_width)
        draw.ellipse(
            [bx - self.bob_radius, by - self.bob_radius,
             bx + self.bob_radius, by + self.bob_radius],
            fill=255,
        )
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=180)  # pivot marker
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

        return np.array(img, dtype=np.uint8)