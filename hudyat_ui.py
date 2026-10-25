"""
hudyat_ui.py - The on-screen interface of record_tool.py.

DISPLAY ONLY. Every function here draws on a DISPLAY COPY of a camera frame
(made by compose_video(), which always returns a new array). The frames that
are written to the video files never pass through this module, so no text,
landmark, button or mirroring can ever end up in a saved clip.

How it stays fast (~30 fps preview):
  * Each UI element (pill, button, card, title, chip, ...) is drawn ONCE with
    Pillow (smooth anti-aliased Poppins text, supersampled rounded shapes) and
    cached as a "Sprite": a small premultiplied-alpha image. A sprite is only
    rebuilt when its content changes (e.g. the text of a pill).
  * Each frame, sprites are alpha-blended onto the display copy only inside
    their own small rectangle, never over the whole frame.
  * Hand landmarks are drawn with OpenCV; the glow is a blur of a small region
    around each hand.

The UI is laid out on a fixed 1280x720 canvas (UI_W x UI_H). The window is
resizable and OpenCV scales the canvas, so the layout scales with it.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import hudyat_common as hc

# =============================================================================
# SETTINGS
# =============================================================================
UI_W, UI_H = 1280, 720          # UI canvas size (the window scales it)
MARGIN = 24                     # distance of UI elements from the canvas edge
SUPERSAMPLE = 3                 # shapes are drawn 3x larger then shrunk = smooth edges
PANEL_ALPHA = 0.55              # opacity of the dark translucent backgrounds
SPRITE_CACHE_SIZE = 800         # max number of cached UI elements

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_FILES = {                  # Poppins, SIL Open Font License (assets/fonts/OFL.txt)
    "regular": "Poppins-Regular.ttf",
    "medium": "Poppins-Medium.ttf",
    "semibold": "Poppins-SemiBold.ttf",
    "bold": "Poppins-Bold.ttf",
}

# Colours as RGB (Pillow). Use bgr() when drawing with OpenCV.
WHITE = (255, 255, 255)
PRIMARY = (47, 107, 255)        # blue: main actions, current item, progress
PRIMARY_LIGHT = (86, 186, 255)  # end of the blue gradient
SUCCESS = (34, 197, 94)         # green
DANGER = (239, 68, 68)          # red: recording
WARNING = (250, 204, 21)        # yellow: warnings
MUTED = (165, 170, 182)         # gray text / inactive
DARK = (10, 12, 18)             # translucent panel background
HAND_COLOR = (255, 40, 40)      # hand landmarks (both hands)

# Head-to-waist corner brackets (fractions of the video area, same region as
# the old framing rectangle in hudyat_common).
GUIDE_BRACKET_LEN = 0.06        # bracket arm length, fraction of the video height
GUIDE_COLOR = (225, 225, 225)


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _rgba(rgb: tuple[int, int, int], alpha: float = 1.0) -> tuple[int, int, int, int]:
    return rgb[0], rgb[1], rgb[2], int(round(alpha * 255))


# =============================================================================
# SPRITES (cached, pre-rendered UI elements)
# =============================================================================
class Sprite:
    """A pre-rendered RGBA element, stored premultiplied for fast blending."""

    __slots__ = ("premult", "inv_alpha", "w", "h", "ox", "oy")

    def __init__(self, rgba: Image.Image, ox: int = 0, oy: int = 0) -> None:
        arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0
        alpha = arr[:, :, 3:4]
        self.premult = arr[:, :, 2::-1] * alpha * 255.0      # RGB -> BGR, premultiplied
        self.inv_alpha = 1.0 - alpha
        self.h, self.w = arr.shape[:2]
        self.ox, self.oy = ox, oy                            # e.g. room for a shadow

    def draw(self, canvas: np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
        """Blend the sprite onto canvas with its top-left corner at (x, y)."""
        x, y = int(round(x)) - self.ox, int(round(y)) - self.oy
        H, W = canvas.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + self.w, W), min(y + self.h, H)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - x, y0 - y
        pm = self.premult[sy:sy + y1 - y0, sx:sx + x1 - x0]
        ia = self.inv_alpha[sy:sy + y1 - y0, sx:sx + x1 - x0]
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        if opacity < 1.0:
            out = roi * (1.0 - (1.0 - ia) * opacity) + pm * opacity
        else:
            out = roi * ia + pm
        canvas[y0:y1, x0:x1] = (out + 0.5).astype(np.uint8)

    @property
    def width(self) -> int:
        return self.w - 2 * self.ox

    @property
    def height(self) -> int:
        return self.h - 2 * self.oy


_sprites: OrderedDict[tuple, Sprite] = OrderedDict()


def cached(key: tuple, build: Callable[[], Image.Image], ox: int = 0, oy: int = 0) -> Sprite:
    """Return the cached sprite for key, building it (once) if needed."""
    sprite = _sprites.get(key)
    if sprite is None:
        sprite = Sprite(build(), ox, oy)
        _sprites[key] = sprite
        if len(_sprites) > SPRITE_CACHE_SIZE:
            _sprites.popitem(last=False)                    # forget the oldest
    else:
        _sprites.move_to_end(key)
    return sprite


@lru_cache(maxsize=64)
def font(size: int, weight: str = "medium") -> ImageFont.FreeTypeFont:
    """Poppins at the given size; falls back to Pillow's built-in font."""
    try:
        return ImageFont.truetype(str(FONT_DIR / FONT_FILES[weight]), size)
    except OSError:
        if not _font_warned:
            _font_warned.append(True)
            print(f"WARNING: font files not found in {FONT_DIR}; using a fallback font.")
        return ImageFont.load_default(size)


_font_warned: list[bool] = []


def text_width(text: str, size: int, weight: str = "medium") -> int:
    return int(math.ceil(font(size, weight).getlength(text)))


def _big(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    """A transparent supersampled layer for drawing smooth shapes."""
    s = SUPERSAMPLE
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img), s


def _shrink(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.Resampling.LANCZOS)


def _draw_icon(d: ImageDraw.ImageDraw, kind: str, x0: float, y0: float, size: float,
               color: tuple[int, int, int, int]) -> None:
    """Vector icons (Poppins has no check / arrow glyphs). Coordinates are on
    the supersampled layer; the icon fills the square (x0, y0, size)."""
    def p(u: float, v: float) -> tuple[float, float]:
        return x0 + u * size, y0 + v * size

    lw = max(2, int(size * 0.10))
    if kind == "back":                                     # chevron  <
        d.line([p(.60, .26), p(.38, .50), p(.60, .74)], fill=color, width=lw, joint="curve")
    elif kind == "check":                                  # tick
        d.line([p(.20, .52), p(.42, .73), p(.80, .30)], fill=color, width=lw, joint="curve")
    elif kind == "retake":                                 # circular arrow (counter-clockwise)
        d.arc([p(.22, .22), p(.78, .78)], start=-30, end=250, fill=color, width=lw)
        cx, cy, r = x0 + .5 * size, y0 + .5 * size, .28 * size
        a = math.radians(-30)
        ex, ey = cx + r * math.cos(a), cy + r * math.sin(a)
        dx, dy = math.sin(a), -math.cos(a)                 # counter-clockwise direction
        px, py = -dy, dx
        hs = size * .16
        d.polygon([(ex + dx * hs, ey + dy * hs), (ex + px * hs * .8, ey + py * hs * .8),
                   (ex - px * hs * .8, ey - py * hs * .8)], fill=color)
    elif kind == "settings":                               # three sliders
        for v, knob in ((.32, .62), (.50, .38), (.68, .56)):
            d.line([p(.22, v), p(.78, v)], fill=color, width=max(2, lw * 2 // 3))
            kx, ky = p(knob, v)
            r = size * .075
            d.ellipse([kx - r, ky - r, kx + r, ky + r], fill=color)
    elif kind == "up":                                     # chevron ^
        d.line([p(.26, .62), p(.50, .38), p(.74, .62)], fill=color, width=lw, joint="curve")
    elif kind == "down":                                   # chevron v
        d.line([p(.26, .38), p(.50, .62), p(.74, .38)], fill=color, width=lw, joint="curve")
    elif kind == "list":                                   # "view takes": three lines with dots
        for v in (.32, .50, .68):
            r = size * .055
            d.ellipse([x0 + .24 * size - r, y0 + v * size - r, x0 + .24 * size + r, y0 + v * size + r], fill=color)
            d.line([p(.38, v), p(.78, v)], fill=color, width=max(2, lw * 2 // 3))
    elif kind == "record":                                 # filled dot
        d.ellipse([p(.24, .24), p(.76, .76)], fill=color)
    elif kind == "pause":                                  # two bars
        d.rounded_rectangle([p(.28, .24), p(.43, .76)], radius=size * .04, fill=color)
        d.rounded_rectangle([p(.57, .24), p(.72, .76)], radius=size * .04, fill=color)
    elif kind == "play":                                   # triangle
        d.polygon([p(.32, .22), p(.32, .78), p(.78, .50)], fill=color)
    elif kind == "skip":                                   # triangle + bar
        d.polygon([p(.24, .24), p(.24, .76), p(.62, .50)], fill=color)
        d.rounded_rectangle([p(.66, .24), p(.78, .76)], radius=size * .03, fill=color)
    elif kind == "flag":                                   # pole + flag
        d.line([p(.28, .18), p(.28, .84)], fill=color, width=lw)
        d.polygon([p(.30, .20), p(.78, .34), p(.30, .50)], fill=color)
    elif kind == "chart":                                  # three bars (overview)
        for u, top in ((.24, .50), (.44, .28), (.64, .40)):
            d.rounded_rectangle([p(u, top), p(u + .13, .78)], radius=size * .03, fill=color)


# =============================================================================
# UI ELEMENTS
# =============================================================================
def pill(text: str, size: int = 16, weight: str = "medium", fg: tuple = WHITE,
         bg: tuple = DARK, bg_alpha: float = PANEL_ALPHA, dot: tuple | None = None,
         dot_alpha: float = 1.0, height: int | None = None, pad: int = 16,
         icon: str | None = None, fg_alpha: float = 1.0) -> Sprite:
    """Fully rounded label: [dot] [icon] text."""
    key = ("pill", text, size, weight, fg, bg, bg_alpha, dot, round(dot_alpha, 2), height, pad, icon, fg_alpha)

    def build() -> Image.Image:
        h = height or int(size * 2.1)
        dot_d = int(h * 0.30) if dot else 0
        icon_s = int(h * 0.55) if icon else 0
        tw = text_width(text, size, weight)
        w = pad * 2 + tw + (dot_d + 9 if dot else 0) + (icon_s + 6 if icon else 0)
        img, d, s = _big(w, h)
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=h * s // 2, fill=_rgba(bg, bg_alpha))
        x = pad
        if dot:
            cy = h / 2
            d.ellipse([x * s, (cy - dot_d / 2) * s, (x + dot_d) * s, (cy + dot_d / 2) * s],
                      fill=_rgba(dot, dot_alpha))
            x += dot_d + 9
        if icon:
            _draw_icon(d, icon, x * s, (h - icon_s) / 2 * s, icon_s * s, _rgba(fg, fg_alpha))
            x += icon_s + 6
        img = _shrink(img, w, h)
        ImageDraw.Draw(img).text((x, h / 2), text, font=font(size, weight),
                                 fill=_rgba(fg, fg_alpha), anchor="lm")
        return img

    return cached(key, build)


def title(text: str, size: int) -> Sprite:
    """Big bold white text with a soft drop shadow (the sign name)."""
    pad = 16

    def build() -> Image.Image:
        f = font(size, "bold")
        asc, desc = f.getmetrics()
        w, h = text_width(text, size, "bold") + 2 * pad + 4, asc + desc + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad + 2, pad + 4), text, font=f, fill=(0, 0, 0, 170))
        img = img.filter(ImageFilter.GaussianBlur(7))
        ImageDraw.Draw(img).text((pad, pad), text, font=f, fill=_rgba(WHITE))
        return img

    return cached(("title", text, size), build, ox=pad, oy=pad)


def fit_title_size(text: str, max_width: int, start: int = 64, minimum: int = 30) -> int:
    """Largest title size (step 4) that fits in max_width."""
    size = start
    while size > minimum and text_width(text, size, "bold") > max_width:
        size -= 4
    return size


def progress_bar(w: int, h: int, fraction: float,
                 colors: tuple[tuple, tuple] = (PRIMARY, PRIMARY_LIGHT)) -> Sprite:
    """Thin rounded bar with a gradient fill."""
    q = int(round(max(0.0, min(1.0, fraction)) * 200))      # 0.5 % steps keep the cache small

    def build() -> Image.Image:
        img, d, s = _big(w, h)
        W, H = w * s, h * s
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 2, fill=(255, 255, 255, 60))
        fw = int(W * q / 200)
        if fw > 0:
            fw = max(fw, H)
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, fw - 1, H - 1), radius=H // 2, fill=255)
            g = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
            c0, c1 = np.array(colors[0], np.float32), np.array(colors[1], np.float32)
            rgb = np.broadcast_to(c0 * (1 - g) + c1 * g, (H, W, 3)).astype(np.uint8)
            img.paste(Image.fromarray(np.ascontiguousarray(rgb)).convert("RGBA"), (0, 0), mask)
        return _shrink(img, w, h)

    return cached(("bar", w, h, q, colors), build)


def round_button(icon: str, d: int, bg: tuple = DARK, bg_alpha: float = PANEL_ALPHA,
                 fg: tuple = WHITE) -> Sprite:
    """Circular icon button."""
    def build() -> Image.Image:
        img, dr, s = _big(d, d)
        dr.ellipse((0, 0, d * s - 1, d * s - 1), fill=_rgba(bg, bg_alpha))
        _draw_icon(dr, icon, d * s * .2, d * s * .2, d * s * .6, _rgba(fg))
        return _shrink(img, d, d)

    return cached(("round", icon, d, bg, bg_alpha, fg), build)


def key_badge(letter: str) -> Sprite:
    """Tiny white key-cap showing a keyboard shortcut (e.g. Q, Esc)."""
    def build() -> Image.Image:
        h = 22
        w = max(22, text_width(letter, 12, "bold") + 12)
        img, d, s = _big(w, h)
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=6 * s, fill=(255, 255, 255, 235))
        img = _shrink(img, w, h)
        ImageDraw.Draw(img).text((w / 2, h / 2), letter, font=font(12, "bold"), fill=_rgba(DARK), anchor="mm")
        return img

    return cached(("badge", letter), build)


def keep_button(w: int, h: int) -> Sprite:
    """Large blue 'Keep (K)' button with a tick icon."""
    def build() -> Image.Image:
        img, d, s = _big(w, h)
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=h * s // 2, fill=_rgba(PRIMARY))
        icon = h * 0.46
        _draw_icon(d, "check", 22 * s, (h - icon) / 2 * s, icon * s, _rgba(WHITE))
        img = _shrink(img, w, h)
        dr = ImageDraw.Draw(img)
        dr.text((22 + icon + 8, h / 2), "Keep", font=font(22, "semibold"), fill=_rgba(WHITE), anchor="lm")
        dr.text((w - 22, h / 2), "(K)", font=font(15, "medium"), fill=(255, 255, 255, 190), anchor="rm")
        return img

    return cached(("keep", w, h), build)


def countdown_badge(n: int, d: int = 220) -> Sprite:
    """Large number inside a translucent circle."""
    def build() -> Image.Image:
        img, dr, s = _big(d, d)
        dr.ellipse((0, 0, d * s - 1, d * s - 1), fill=_rgba(DARK, 0.5),
                   outline=(255, 255, 255, 90), width=3 * s)
        img = _shrink(img, d, d)
        ImageDraw.Draw(img).text((d / 2, d / 2), str(n), font=font(120, "bold"), fill=_rgba(WHITE), anchor="mm")
        return img

    return cached(("count", n, d), build)


def chip(text: str, state: str, max_chars: int = 22) -> Sprite:
    """Sign-list chip. state: 'done' | 'current' | 'next' | 'more'."""
    if len(text) > max_chars:
        text = text[:max_chars - 1] + "…"
    if state == "done":
        return pill(text, 14, "medium", WHITE, DARK, 0.50, dot=SUCCESS, height=30, pad=12,
                    icon="check", fg_alpha=0.85)
    if state == "current":
        return pill(text, 14, "semibold", WHITE, PRIMARY, 0.92, dot=WHITE, height=30, pad=12)
    if state == "more":
        return pill(text, 13, "regular", MUTED, DARK, 0.30, height=26, pad=12)
    return pill(text, 14, "regular", MUTED, DARK, 0.35, height=30, pad=12)


def card(rows: list[tuple[str, str, tuple, bool]], heading: str, width: int = 330,
         tag: str | None = None) -> Sprite:
    """Rounded translucent card: heading, then (label, value, value colour,
    highlighted) rows. A highlighted row gets a blue tint and the tag text."""
    key = ("card", tuple(rows), heading, width, tag)

    def build() -> Image.Image:
        row_h, top = 36, 54
        h = top + row_h * len(rows) + 14
        img, d, s = _big(width, h)
        d.rounded_rectangle((0, 0, width * s - 1, h * s - 1), radius=18 * s, fill=_rgba(DARK, 0.62))
        for i, (_, _, _, hi) in enumerate(rows):
            if hi:
                y = top + i * row_h
                d.rounded_rectangle((10 * s, (y + 2) * s, (width - 10) * s, (y + row_h - 2) * s),
                                    radius=10 * s, fill=_rgba(PRIMARY, 0.35), outline=_rgba(PRIMARY, 0.9), width=s)
        img = _shrink(img, width, h)
        dr = ImageDraw.Draw(img)
        dr.text((20, 30), heading, font=font(18, "semibold"), fill=_rgba(WHITE), anchor="lm")
        for i, (label, value, color, hi) in enumerate(rows):
            y = top + i * row_h + row_h / 2
            dr.text((22, y), label, font=font(15, "regular"), fill=_rgba(WHITE if hi else MUTED), anchor="lm")
            if hi and tag:
                lx = 22 + text_width(label, 15, "regular") + 8
                dr.text((lx, y), tag, font=font(11, "semibold"), fill=_rgba(PRIMARY_LIGHT), anchor="lm")
            dr.text((width - 22, y), value, font=font(17, "semibold"), fill=_rgba(color), anchor="rm")
        return img

    return cached(key, build)


# A text item for box(): (x, y, text, size, weight, rgb, alpha, anchor)
TextItem = tuple[float, float, str, int, str, tuple, float, str]
# An icon item for box(): (kind, x, y, size, rgb, alpha)
IconItem = tuple[str, float, float, float, tuple, float]


def box(w: int, h: int, radius: int = 18, bg: tuple = DARK, bg_alpha: float = 0.6,
        outline: tuple | None = None, outline_alpha: float = 0.9, outline_w: int = 2,
        texts: tuple[TextItem, ...] = (), icons: tuple[IconItem, ...] = ()) -> Sprite:
    """Generic rounded card with any number of text and icon items (cached).

    Used for the signer cards, category tiles, sign / take rows and the
    overview table cells. Anchors are Pillow anchors, e.g. "lm" = left-middle.
    """
    key = ("box", w, h, radius, bg, round(bg_alpha, 3), outline, outline_alpha, outline_w, texts, icons)

    def build() -> Image.Image:
        img, d, s = _big(w, h)
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=radius * s, fill=_rgba(bg, bg_alpha),
                            outline=_rgba(outline, outline_alpha) if outline else None,
                            width=outline_w * s if outline else 0)
        for kind, x, y, size, rgb, alpha in icons:
            _draw_icon(d, kind, x * s, y * s, size * s, _rgba(rgb, alpha))
        img = _shrink(img, w, h)
        dr = ImageDraw.Draw(img)
        for x, y, text, size, weight, rgb, alpha, anchor in texts:
            dr.text((x, y), text, font=font(size, weight), fill=_rgba(rgb, alpha), anchor=anchor)
        return img

    return cached(key, build)


def hover_outline(w: int, h: int, radius: int) -> Sprite:
    """Thin white outline drawn over the clickable item under the mouse."""
    return box(w, h, radius, DARK, 0.0, outline=WHITE, outline_alpha=0.55, outline_w=2)


_backdrop_fallback: list[np.ndarray] = []


def backdrop(frame: np.ndarray | None) -> np.ndarray:
    """Dashboard background: the live camera, mirrored, blurred and darkened.

    Works on a tiny copy, so it is cheap; returns a NEW UI_W x UI_H canvas.
    """
    if frame is None:
        if not _backdrop_fallback:
            g = np.linspace(28, 12, UI_H, dtype=np.float32)[:, None, None]
            _backdrop_fallback.append(np.broadcast_to(g, (UI_H, UI_W, 3)).astype(np.uint8))
        return _backdrop_fallback[0].copy()
    small = cv2.resize(cv2.flip(frame, 1), (160, 90), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    canvas = cv2.resize(small, (UI_W, UI_H), interpolation=cv2.INTER_LINEAR)
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=0.38, beta=6)
    return canvas


# =============================================================================
# VIDEO, GUIDES, HANDS
# =============================================================================
@dataclass
class VideoRect:
    """Where the camera image sits on the UI canvas."""
    x: int
    y: int
    w: int
    h: int


def compose_video(frame: np.ndarray, mirror: bool = True) -> tuple[np.ndarray, VideoRect]:
    """Return a NEW UI_W x UI_H display canvas showing the frame.

    cv2.flip / cv2.resize / np.zeros all allocate new arrays, so the original
    frame (which may also be on its way to the video writer) is never changed.
    """
    display = cv2.flip(frame, 1) if mirror else frame.copy()
    fh, fw = display.shape[:2]
    if (fw, fh) == (UI_W, UI_H):
        return display, VideoRect(0, 0, UI_W, UI_H)
    scale = min(UI_W / fw, UI_H / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas = np.zeros((UI_H, UI_W, 3), np.uint8)
    x0, y0 = (UI_W - nw) // 2, (UI_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(display, (nw, nh), interpolation=interp)
    return canvas, VideoRect(x0, y0, nw, nh)


def blank_canvas() -> tuple[np.ndarray, VideoRect]:
    return np.full((UI_H, UI_W, 3), 18, np.uint8), VideoRect(0, 0, UI_W, UI_H)


def dim(canvas: np.ndarray, amount: float = 0.45) -> None:
    """Darken the whole display (pause / busy screens)."""
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=1.0 - amount)


def draw_guides(canvas: np.ndarray, rect: VideoRect) -> None:
    """Subtle corner brackets marking the head-to-waist area (no face shapes)."""
    x0 = int(rect.x + hc.GUIDE_LEFT * rect.w)
    x1 = int(rect.x + hc.GUIDE_RIGHT * rect.w)
    y0 = int(rect.y + hc.GUIDE_TOP * rect.h)
    y1 = int(rect.y + hc.GUIDE_BOTTOM * rect.h)
    arm = int(GUIDE_BRACKET_LEN * rect.h)
    color = bgr(GUIDE_COLOR)
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(canvas, (cx, cy), (cx + dx * arm, cy), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * arm), color, 2, cv2.LINE_AA)
    mid = rect.x + rect.w // 2                                  # tiny centre ticks
    cv2.line(canvas, (mid, y0), (mid, y0 + 12), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (mid, y1), (mid, y1 - 12), color, 1, cv2.LINE_AA)


FINGERTIPS = {4, 8, 12, 16, 20}
_LM_COLOR = [bgr(HAND_COLOR)] * 21               # colour of each of the 21 landmarks


def draw_hands(canvas: np.ndarray, hands: list[hc.DetectedHand], rect: VideoRect,
               mirrored: bool = True, glow: bool = True, labels: bool = True) -> None:
    """Hand skeletons: glowing red dots + thin red lines (same colour for both hands).

    Landmarks are in UNMIRRORED image coordinates; on the mirrored display,
    x becomes 1 - x. Labels show the signer's REAL hand (Left / Right).
    """
    H, W = canvas.shape[:2]
    for hand in hands:
        xs = 1.0 - hand.landmarks[:, 0] if mirrored else hand.landmarks[:, 0]
        pts = [(int(rect.x + x * rect.w), int(rect.y + y * rect.h)) for x, y in zip(xs, hand.landmarks[:, 1])]
        if glow:                                       # soft glow: blurred copy, added in a small region
            pad = 26
            bx0 = max(min(p[0] for p in pts) - pad, 0)
            by0 = max(min(p[1] for p in pts) - pad, 0)
            bx1 = min(max(p[0] for p in pts) + pad, W)
            by1 = min(max(p[1] for p in pts) + pad, H)
            if bx1 > bx0 and by1 > by0:
                layer = np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8)
                local = [(px - bx0, py - by0) for px, py in pts]
                for a, b in hc.HAND_CONNECTIONS:
                    cv2.line(layer, local[a], local[b], _LM_COLOR[b], 6, cv2.LINE_AA)
                for i, p in enumerate(local):
                    cv2.circle(layer, p, 9 if i in FINGERTIPS else 7, _LM_COLOR[i], -1, cv2.LINE_AA)
                layer = cv2.GaussianBlur(layer, (0, 0), 6)
                roi = canvas[by0:by1, bx0:bx1]
                cv2.addWeighted(roi, 1.0, layer, 0.7, 0, dst=roi)
        for a, b in hc.HAND_CONNECTIONS:
            col = tuple((ca + cb) // 2 for ca, cb in zip(_LM_COLOR[a], _LM_COLOR[b]))
            cv2.line(canvas, pts[a], pts[b], col, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            tip = i in FINGERTIPS
            cv2.circle(canvas, p, 5 if tip else 3, _LM_COLOR[i], -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 2 if tip else 1, (255, 255, 255), -1, cv2.LINE_AA)
        if labels:
            tag = pill(hand.label, 12, "semibold", WHITE, DARK, 0.6, height=22, pad=9)
            tag.draw(canvas, pts[0][0] - tag.width // 2, pts[0][1] + 14)


def brightness_info(value: float) -> tuple[str, tuple, bool]:
    """(pill text, dot colour, is_warning) for the brightness indicator."""
    if value < hc.BRIGHTNESS_TOO_DARK:
        return f"Too dark ({value:.0f}) — add front light", WARNING, True
    if value > hc.BRIGHTNESS_TOO_BRIGHT:
        return f"Too bright ({value:.0f}) — reduce light", WARNING, True
    return f"Light OK ({value:.0f})", SUCCESS, False


def pretty_category(category: str) -> str:
    """'wh_questions' -> 'Wh Questions'"""
    return category.replace("_", " ").title()


# =============================================================================
# CLICKABLE REGIONS (mouse support for the on-screen buttons)
# =============================================================================
class HitMap:
    """Rectangles registered while drawing; a click returns the matching key."""

    def __init__(self) -> None:
        self._areas: list[tuple[int, int, int, int, str]] = []

    def clear(self) -> None:
        self._areas.clear()

    def add(self, x: int, y: int, w: int, h: int, key: str) -> None:
        self._areas.append((x, y, x + w, y + h, key))

    def key_at(self, x: int, y: int) -> str | None:
        area = self.area_at(x, y)
        return area[4] if area else None

    def area_at(self, x: int, y: int) -> tuple[int, int, int, int, str] | None:
        """(x0, y0, x1, y1, key) of the top-most area under the point."""
        for area in reversed(self._areas):
            x0, y0, x1, y1, _ = area
            if x0 <= x <= x1 and y0 <= y <= y1:
                return area
        return None
