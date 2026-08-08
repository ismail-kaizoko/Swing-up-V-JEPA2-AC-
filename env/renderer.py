"""
Rasterizes a pendulum angle into a small square grayscale image. Uses PIL's
ImageDraw rather than matplotlib — roughly 50-100x faster per frame, which
matters once you're generating tens of thousands of frames for pretraining.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image, ImageDraw
import numpy as np
# import pygame
# from pygame import gfxdraw


class PendulumRenderer:
    def __init__(self, img_size = 64, rod_length_prop: int =0.45,
                 rod_width_prop: int = 0.05, bob_radius_prop: int = 0.05):
        self.img_size = img_size
        self.center = tuple(img_size*np.array((0.5 , 0.3)))
        self.rod_length = int(img_size*rod_length_prop)
        self.rod_width = int(img_size*rod_width_prop)
        self.bob_radius = int(img_size*bob_radius_prop)
        # self.surf = pygame.Surface((img_size, img_size))

    def render(self, theta: float) -> np.ndarray:
        img = Image.new("L", (self.img_size, self.img_size), color=0)  # "L" = 8-bit grayscale
        draw = ImageDraw.Draw(img)
        cx, cy = tuple(int(x) for x in self.center)

        # theta=0 is upright: in image coordinates y grows *downward*, so
        # the bob offset from the pivot is (sin(theta), -cos(theta)) —
        # at theta=0 that's (0, -1): straight up on screen, as intended.
        bx = cx + self.rod_length * np.sin(theta)
        by = cy - self.rod_length * np.cos(theta)

        draw.line([(cx, cy), (bx, by)], fill=255, width=self.rod_width)
        draw.ellipse(
            [bx - self.bob_radius, by - self.bob_radius,
             bx + self.bob_radius, by + self.bob_radius],
            fill=255,
        )
        draw.ellipse([cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5], fill=180)  # pivot marker
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

        return np.array(img, dtype=np.uint8)

    # def render(self, theta: float) -> np.ndarray:
    #     self.surf.fill((255, 255, 255))     # white background
    #     cx, cy = self.center

    #     dx, dy = np.sin(theta), -np.cos(theta)
    #     px, py = -dy, dx
    #     hw = self.rod_width / 2
    #     tip_x, tip_y = cx + self.rod_length * dx, cy + self.rod_length * dy

    #     rod_points = [
    #         (cx + px * hw, cy + py * hw),
    #         (tip_x + px * hw, tip_y + py * hw),
    #         (tip_x - px * hw, tip_y - py * hw),
    #         (cx - px * hw, cy - py * hw),
    #     ]
    #     gfxdraw.aapolygon(self.surf, rod_points, (0, 0, 0))
    #     gfxdraw.filled_polygon(self.surf, rod_points, (0, 0, 0))

    #     # rounded ends, same color/radius as rod half-width -> reads as one
    #     # capsule-shaped rod, not a rectangle with a separate ball on top
    #     for x, y in [(cx, cy), (tip_x, tip_y)]:
    #         gfxdraw.aacircle(self.surf, round(x), round(y), round(hw), (0, 0, 0))
    #         gfxdraw.filled_circle(self.surf, round(x), round(y), round(hw), (0, 0, 0))

    #     arr = pygame.surfarray.array3d(self.surf)
    #     arr = np.transpose(arr, (1, 0, 2))
    #     return arr[:, :, 0]                   # channels are identical (only drew white/gray) -> take one as grayscale