#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Effects GUI — v7 (UI stretch + big speed-ups)

Nouveautés vs v6
----------------
UI
- Le panneau **Configure** s'étire jusqu'au bord de la fenêtre (plus large).
- Colonne gauche et bloc paramètres **responsive** (fill="both", expand=True).
- Molette + scrollbars fiables.

Perf
- Prévisualisation asynchrone (worker thread) pour ne jamais bloquer l’UI.
- Dither Bayer vectorisé (x50+ plus rapide que la double boucle Python).
- Accélération **CUDA/OpenCV** automatique si disponible (checkbox GPU). Fallback CPU sûr.
- Export vidéo : option **NVENC** (s’il est présent dans ffmpeg), choix *preset* et *threads*.
- Redimensionnement vidéo/image optimisé (OpenCV, pas Pillow) côté preview.

Dépendances : numpy, pillow, opencv-python, moviepy, imageio-ffmpeg
(identiques à avant — CUDA est optionnel ; si non dispo, tout marche en CPU)
"""

import os, math, threading, queue, json, copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
import cv2

# MoviePy imports (2.x avec fallback 1.x)
try:
    from moviepy import VideoFileClip
except Exception:  # pragma: no cover
    from moviepy.editor import VideoFileClip  # type: ignore

# ===================== Utils I/O =====================

def imread_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".webp"]:
        return Image.open(path).convert("RGBA")
    return Image.open(path).convert("RGB")

def pil_to_np(img):
    return np.array(img)

def np_to_pil(arr, keep_alpha=False):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if keep_alpha and arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr, mode="RGB")

def has_alpha(img_pil):
    return img_pil.mode == "RGBA"

def ensure_three_channels(frame):
    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)
    if frame.shape[-1] == 4:
        return frame[:, :, :3]
    return frame

# ===================== CUDA helpers =====================

def cuda_available():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

USE_CUDA_DEFAULT = cuda_available()

# wrappers simples (si CUDA dispo) -----------------------------------------

def cuda_gaussian_blur(src, ksize, sigma):
    if not USE_CUDA_DEFAULT:
        return cv2.GaussianBlur(src, ksize, sigma, borderType=cv2.BORDER_REPLICATE)
    try:
        g = cv2.cuda_GpuMat(); g.upload(src)
        blurred = cv2.cuda.createGaussianFilter(g.type(), g.type(), ksize, sigma).apply(g)
        return blurred.download()
    except Exception:
        return cv2.GaussianBlur(src, ksize, sigma, borderType=cv2.BORDER_REPLICATE)

def cuda_warp_affine(src, M, size):
    if not USE_CUDA_DEFAULT:
        return cv2.warpAffine(src, M, size, borderMode=cv2.BORDER_REFLECT)
    try:
        g = cv2.cuda_GpuMat(); g.upload(src)
        out = cv2.cuda.warpAffine(g, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out.download()
    except Exception:
        return cv2.warpAffine(src, M, size, borderMode=cv2.BORDER_REFLECT)

def cuda_filter2d(src, kernel):
    if not USE_CUDA_DEFAULT:
        return cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    try:
        g = cv2.cuda_GpuMat(); g.upload(src)
        filt = cv2.cuda.createLinearFilter(g.type(), g.type(), kernel)
        out = filt.apply(g)
        return out.download()
    except Exception:
        return cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# ===================== Couleur / NTSC =====================

def rgb_to_yiq(rgb):
    R = rgb[:, :, 0]; G = rgb[:, :, 1]; B = rgb[:, :, 2]
    Y = 0.299*R + 0.587*G + 0.114*B
    I = 0.596*R - 0.275*G - 0.321*B
    Q = 0.212*R - 0.523*G + 0.311*B
    return Y, I, Q

def yiq_to_rgb(Y, I, Q):
    R = Y + 0.956*I + 0.621*Q
    G = Y - 0.272*I - 0.647*Q
    B = Y - 1.106*I + 1.703*Q
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)

def adjust_saturation_iq(I, Q, factor):
    if factor == 1.0:
        return I, Q
    return I*factor, Q*factor

def unsharp_luma(Y, amount=0.0, radius=1.2):
    if amount <= 0:
        return Y
    k = max(3, int(2*round(radius)+1))
    blur = cuda_gaussian_blur(Y, (k, k), radius)
    return np.clip(Y + amount*(Y - blur) + 0.25*amount*np.sign(Y - blur), 0, 255)

def horiz_smear(img, radius):
    r = int(round(radius))
    if r <= 0:
        return img
    kernel = np.ones((1, 2*r+1), np.float32) / (2*r+1)
    return cuda_filter2d(img, kernel)

def chromatic_fringes(img, strength_px=0.0):
    s = float(strength_px)
    if s <= 0:
        return img
    h, w, _ = img.shape
    out = np.empty_like(img)
    shiftR = int(round(+s))
    shiftB = int(round(-s))
    out[:, :, 2] = cuda_warp_affine(img[:, :, 2], np.float32([[1,0,shiftR],[0,1,0]]), (w, h))
    out[:, :, 1] = img[:, :, 1]
    out[:, :, 0] = cuda_warp_affine(img[:, :, 0], np.float32([[1,0,shiftB],[0,1,0]]), (w, h))
    return out

def add_noise(img, std):
    if std <= 0:
        return img
    noise = np.random.normal(0.0, std, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# ----------------- NTSC main --------------------

def ntsc_with_params(frame_rgb_u8, ringing=20, encoding=1, noise=0.01, saturation=2.0, smear=0.0, fringes=1.0):
    f = ensure_three_channels(frame_rgb_u8).astype(np.float32)
    Y, I, Q = rgb_to_yiq(f)
    I, Q = adjust_saturation_iq(I, Q, float(saturation))

    enc = int(round(encoding))
    sub = [1, 2, 4, 4][enc]
    blur_sigma = [0, 3, 5, 7][enc]
    shift = [0, 1, 2, 3][enc]

    if blur_sigma > 0:
        k = max(3, int(2*round(blur_sigma)+1))
        I = cuda_gaussian_blur(I, (k, 1), blur_sigma)
        Q = cuda_gaussian_blur(Q, (k, 1), blur_sigma)

    if sub > 1:
        h, w = I.shape
        nw = max(1, w//sub)
        I = cv2.resize(I, (nw, h), interpolation=cv2.INTER_AREA)
        Q = cv2.resize(Q, (nw, h), interpolation=cv2.INTER_AREA)
        I = cv2.resize(I, (w, h), interpolation=cv2.INTER_LINEAR)
        Q = cv2.resize(Q, (w, h), interpolation=cv2.INTER_LINEAR)

    if shift:
        I = cuda_warp_affine(I, np.float32([[1,0,shift],[0,1,0]]), (I.shape[1], I.shape[0]))
        Q = cuda_warp_affine(Q, np.float32([[1,0,shift],[0,1,0]]), (Q.shape[1], Q.shape[0]))

    rgb = yiq_to_rgb(Y, I, Q)

    Y2, _, _ = rgb_to_yiq(rgb)
    Y2 = unsharp_luma(Y2, amount=float(ringing)/40.0, radius=1.5)
    rgb = yiq_to_rgb(Y2, I, Q)

    rgb = horiz_smear(rgb, smear)
    rgb = add_noise(rgb.astype(np.uint8), std=float(noise)*255.0)
    rgb = chromatic_fringes(rgb, strength_px=fringes)

    return np.clip(rgb, 0, 255).astype(np.uint8)

# ===================== Dither helpers =====================

def bayer_matrix(n=8):
    if n == 8:
        M = np.array([
            [0,48,12,60,3,51,15,63],
            [32,16,44,28,35,19,47,31],
            [8,56,4,52,11,59,7,55],
            [40,24,36,20,43,27,39,23],
            [2,50,14,62,1,49,13,61],
            [34,18,46,30,33,17,45,29],
            [10,58,6,54,9,57,5,53],
            [42,26,38,22,41,25,37,21]
        ], dtype=np.float32) / 64.0
    else:
        M = np.array([
            [0,8,2,10],
            [12,4,14,6],
            [3,11,1,9],
            [15,7,13,5]
        ], dtype=np.float32) / 16.0
    return M

# Vectorisation massive du Bayer (aucune boucle Python) ---------------------

def bayer_threshold_rgb(base_rgb_u8, size=8):
    H, W, C = base_rgb_u8.shape
    M = bayer_matrix(8 if size >= 8 else 4)
    mh, mw = M.shape
    T = np.tile(M, (int(np.ceil(H/mh)), int(np.ceil(W/mw))))[:H, :W]
    T = (T*255.0).astype(np.float32)
    base = base_rgb_u8.astype(np.float32)
    # Décalage fin pour éviter le banding (équivalent à "grain de seuil")
    thr = base + (T[..., None]-127.5)*0.25
    out = (thr >= T[..., None]).astype(np.uint8)*255
    return out

# Quantize / palettes -------------------------------------------------------

def hsv12(idx, sat=1.0, val=1.0):
    import colorsys
    i = int(idx) % 12
    h = i / 12.0
    r, g, b = colorsys.hsv_to_rgb(h, max(0, min(1, sat)), max(0, min(1, val)))
    return np.array([r*255.0, g*255.0, b*255.0], dtype=np.float32)

def quantize_palette(arr, colors=6, mode=1):
    if mode == 0:
        rgb = ensure_three_channels(arr).astype(np.float32)
        Y = 0.2126*rgb[:, :, 0] + 0.7152*rgb[:, :, 1] + 0.0722*rgb[:, :, 2]
        bins = np.linspace(0, 255, max(2, int(colors)), endpoint=True)
        idx = np.digitize(Y, bins) - 1
        idx = np.clip(idx, 0, len(bins)-1)
        pal = (bins[idx][:, :, None]).repeat(3, axis=2)
        return pal.astype(np.uint8)
    else:
        pil = Image.fromarray(ensure_three_channels(arr).astype(np.uint8))
        q = pil.quantize(colors=max(2, min(256, int(colors))), method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        return np.array(q.convert("RGB"), dtype=np.uint8)

# Pixelation rapide ---------------------------------------------------------

def pixelate_np(arr, k):
    if k <= 1:
        return arr
    h, w = arr.shape[:2]
    nh, nw = max(1, h//k), max(1, w//k)
    small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# Effet Dither (site-like) --------------------------------------------------

def dither_effect_like_site(img_np_u8, p):
    arr = ensure_three_channels(img_np_u8).astype(np.float32)

    g = float(p.get("gamma", 1.0))
    if g != 1.0 and g > 0:
        arr = (np.clip(arr/255.0, 0, 1)**(1.0/g))*255.0
    arr = pixelate_np(arr.astype(np.uint8), int(p.get("pixelStep", 1))).astype(np.float32)

    paltype = int(p.get("palette_type", 2))
    if paltype == 0:
        base = quantize_palette(arr.astype(np.uint8), colors=2, mode=0)
    elif paltype == 1:
        steps = np.array([7, 7, 3], dtype=np.float32)
        q = np.round(arr/255.0*steps)/steps*255.0
        base = q.astype(np.uint8)
    else:
        gb = np.array([[15,56,15],[48,98,48],[139,172,15],[155,188,15]], dtype=np.float32)
        Y = 0.2126*arr[:, :, 0] + 0.7152*arr[:, :, 1] + 0.0722*arr[:, :, 2]
        bins = np.linspace(0, 255, 4, endpoint=True)
        idx = np.digitize(Y, bins) - 1
        idx = np.clip(idx, 0, 3)
        base = gb[idx].astype(np.uint8)

    cc = int(p.get("color_count", 18))
    dm = int(p.get("distance_mode", 1))
    if cc and cc >= 2:
        base = quantize_palette(base, colors=cc, mode=dm)

    pat = int(p.get("pattern_type", 0))
    if pat == 0:
        size = int(p.get("bayer_size", 8))
        dithered = bayer_threshold_rgb(base, size=size)
    else:
        # FSD/Atkinson CPU (assez rapide car entrée déjà quantized)
        dithered = error_diffusion(base, "floyd" if pat == 1 else "atkinson")

    if paltype == 0:
        c1 = hsv12(int(p.get("color1", 8)))
        c2 = hsv12(int(p.get("color2", 11)))
        mask = (np.mean(dithered, axis=2) > 127).astype(np.float32)[..., None]
        dithered = (mask*c1 + (1.0-mask)*c2).astype(np.uint8)

    strength = float(p.get("dither_strength", 2.0))/3.0
    strength = max(0.0, min(1.0, strength))
    mixed = (strength*dithered.astype(np.float32) + (1.0-strength)*arr).astype(np.uint8)
    return mixed

# (réutilisé par Dither FSD/Atkinson)

def error_diffusion(img_u8, method="floyd"):
    a = ensure_three_channels(img_u8).astype(np.float32).copy()
    H, W, _ = a.shape
    if method == "floyd":
        kernel = [(1,0,7/16), (-1,1,3/16), (0,1,5/16), (1,1,1/16)]
    else:
        kernel = [(1,0,1/8), (2,0,1/8), (-1,1,1/8), (0,1,1/8), (1,1,1/8), (0,2,1/8)]
    for y in range(H):
        for x in range(W):
            old = a[y, x].copy()
            new = np.where(old > 127.5, 255.0, 0.0)
            a[y, x] = new
            err = old - new
            for dx, dy, w in kernel:
                xx, yy = x+dx, y+dy
                if 0 <= xx < W and 0 <= yy < H:
                    a[yy, xx] += err*w
    return np.clip(a, 0, 255).astype(np.uint8)

# ===================== Nouveaux filtres =====================

def grayscale_effect(arr_u8, intensity=1.0):
    """intensity 0..1 : 0=aucun, 1=plein gris"""
    arr = ensure_three_channels(arr_u8).astype(np.float32)
    lum = 0.299*arr[:, :, 0] + 0.587*arr[:, :, 1] + 0.114*arr[:, :, 2]
    gray3 = np.stack([lum, lum, lum], axis=-1)
    out = (arr*(1.0-float(intensity)) + gray3*float(intensity))
    return np.clip(out, 0, 255).astype(np.uint8)

def sepia_effect(arr_u8, strength=1.0):
    """strength 0..1"""
    arr = ensure_three_channels(arr_u8).astype(np.float32)
    M = np.array([[0.393,0.769,0.189],
                  [0.349,0.686,0.168],
                  [0.272,0.534,0.131]], dtype=np.float32)
    sep = np.tensordot(arr, M.T, axes=([2],[1]))
    out = arr*(1.0-float(strength)) + sep*float(strength)
    return np.clip(out, 0, 255).astype(np.uint8)

def posterize_effect(arr_u8, levels=4):
    """levels >=2"""
    arr = ensure_three_channels(arr_u8).astype(np.float32)
    l = max(2, int(levels))
    factor = 256.0 / l
    out = (np.floor(arr / factor) * factor + factor/2.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def sobel_edge_effect(arr_u8, strength=1.0):
    """strength 0..1 : blend edges (white) on top"""
    img = ensure_three_channels(arr_u8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255.0
    edge = np.stack([mag, mag, mag], axis=-1)
    out = img.astype(np.float32)*(1.0-float(strength)) + edge*float(strength)
    return np.clip(out, 0, 255).astype(np.uint8)

def vignette_effect(arr_u8, strength=0.8, radius=0.75):
    """strength 0..1, radius ~0.2..2.0 (relative)"""
    arr = ensure_three_channels(arr_u8).astype(np.float32)
    h, w = arr.shape[:2]
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    cx, cy = 0.5, 0.5
    xv = (x - cx) ** 2
    yv = (y - cy) ** 2
    dist = np.sqrt(xv + yv)
    # radius controls where attenuation starts; normalize dist by radius
    r = max(0.001, float(radius))
    mask = 1.0 - np.clip((dist / r) ** 2, 0.0, 1.0)
    mask = (mask * (1.0 - float(strength))) + (mask * float(strength))
    mask3 = mask[:, :, None]
    out = arr * mask3
    return np.clip(out, 0, 255).astype(np.uint8)

# ===================== Effets / pile =====================

def apply_effect(img_np_u8, eff: dict):
    if not eff.get("enabled", True):
        return img_np_u8
    t = eff["type"]; p = eff["params"]
    if t == "ntsc":
        return ntsc_with_params(
            img_np_u8,
            ringing=float(p.get("ringing", 20)),
            encoding=int(p.get("encoding", 1)),
            noise=float(p.get("noise", 0.01)),
            saturation=float(p.get("saturation", 2.0)),
            smear=float(p.get("smear", 0.0)),
            fringes=float(p.get("fringes", 1.0)),
        )
    elif t == "dither":
        return dither_effect_like_site(img_np_u8, p)
    elif t == "gray":
        return grayscale_effect(img_np_u8, strength := float(p.get("gray_strength", 1.0)))
    elif t == "sepia":
        return sepia_effect(img_np_u8, float(p.get("sepia_strength", 1.0)))
    elif t == "posterize":
        return posterize_effect(img_np_u8, int(p.get("posterize_levels", 4)))
    elif t == "edge":
        return sobel_edge_effect(img_np_u8, float(p.get("edge_strength", 1.0)))
    elif t == "vignette":
        return vignette_effect(img_np_u8, float(p.get("vignette_strength", 0.8)), float(p.get("vignette_radius", 0.75)))
    return img_np_u8

def apply_stack_to_pil(img_pil, stack):
    arr = pil_to_np(img_pil)
    out = arr
    for e in stack:
        out = apply_effect(out, e)
    return np_to_pil(out, keep_alpha=has_alpha(img_pil))

def apply_stack_to_frame(frame_np, stack):
    out = frame_np
    for e in stack:
        out = apply_effect(out, e)
    return ensure_three_channels(out)

# ===================== MoviePy cross-version =====================

def apply_framewise(clip, frame_func):
    if hasattr(clip, "fl_image"):
        try:
            return clip.fl_image(frame_func)
        except Exception:
            pass
    try:
        from moviepy.video.fx.all import fl_image as vfx_fl_image
        return clip.fx(vfx_fl_image, frame_func)
    except Exception:
        pass
    make = clip.make_frame
    def new_make_frame(t):
        return frame_func(make(t))
    return clip.set_make_frame(new_make_frame)

# ===================== GUI =====================

DEFAULT_CFG = dict(
    # commun NTSC / Dither
    ringing=20,
    encoding=1,
    noise=0.01,
    saturation=2.0,
    smear=0.0,
    fringes=1.0,
    # Dither
    pattern_type=0,
    palette_type=2,
    color_count=18,
    distance_mode=1,
    dither_strength=2,
    gamma=1.6,
    pixelStep=2,
    bayer_size=8,
    color1=8,
    color2=11,
    # Nouveaux filtres defaults
    gray_strength=1.0,
    sepia_strength=1.0,
    posterize_levels=4,
    edge_strength=1.0,
    vignette_strength=0.8,
    vignette_radius=0.75,
)

class EffectsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Effects Preview — v7")
        self.geometry("1380x840"); self.minsize(1200, 760)

        self.input_path=None; self.input_is_video=False
        self.loaded_image=None; self.video_clip=None; self.video_duration=0.0

        self.stack=[]
        self.preview_imgtk=None
        self.live_preview=tk.BooleanVar(value=True)
        self.use_cuda=tk.BooleanVar(value=USE_CUDA_DEFAULT)
        self.use_nvenc=tk.BooleanVar(value=False)
        self.ffmpeg_preset=tk.StringVar(value="medium")
        self.ffmpeg_threads=tk.IntVar(value=max(2, (os.cpu_count() or 4)//2))

        self._render_pending=False
        self._render_q = queue.Queue(maxsize=1)  # garder la dernière demande seulement
        self._render_thread = threading.Thread(target=self._render_worker, daemon=True)
        self._render_thread.start()

        self._build_ui()

    # ---------- Layout ----------
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Colonne gauche large et responsive
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsw")
        left.rowconfigure(3, weight=1)  # le bloc paramètres s'étire

        # Fichier
        fb = ttk.LabelFrame(left, text="Fichier"); fb.grid(row=0, column=0, sticky="new", padx=8, pady=6)
        fb.columnconfigure(0, weight=1)
        ttk.Button(fb, text="Ouvrir image/vidéo...", command=self.on_open).grid(row=0, column=0, sticky="ew", padx=8, pady=(8,4))
        # boutons config (.FXOption)
        brow = ttk.Frame(fb); brow.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,6))
        ttk.Button(brow, text="Sauvegarder config (.FXOption)", command=self.on_save_config).pack(side="left", expand=False)
        ttk.Button(brow, text="Ouvrir config (.FXOption/.json)", command=self.on_load_config).pack(side="left", padx=6)
        self.info_label = ttk.Label(fb, text="Aucun fichier chargé")
        self.info_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0,8))

        # Aperçu
        pv = ttk.LabelFrame(left, text="Aperçu"); pv.grid(row=1, column=0, sticky="new", padx=8, pady=6)
        pv.columnconfigure(0, weight=1)
        ttk.Checkbutton(pv, text="Aperçu en temps réel", variable=self.live_preview).grid(row=0, column=0, sticky="w", padx=8, pady=(8,2))
        ttk.Button(pv, text="Actualiser maintenant", command=self.request_render).grid(row=1, column=0, sticky="ew", padx=8, pady=(4,8))

        # Vidéo
        self.video_box = ttk.LabelFrame(left, text="Vidéo"); self.video_box.grid(row=2, column=0, sticky="new", padx=8, pady=6)
        self.video_box.columnconfigure(0, weight=1)
        ttk.Label(self.video_box, text="Position (s)").grid(row=0, column=0, sticky="w", padx=8, pady=(8,0))
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub = ttk.Scale(self.video_box, from_=0, to=1, variable=self.scrub_var, orient="horizontal", command=lambda *_: self._maybe_render())
        self.scrub.grid(row=1, column=0, sticky="ew", padx=8, pady=(2,8))
        self.video_box.grid_remove()

        # Pile d’effets
        sb = ttk.LabelFrame(left, text="Pile d'effets (haut -> bas)"); sb.grid(row=3, column=0, sticky="new", padx=8, pady=6)
        sb.columnconfigure(0, weight=1)
        self.stack_list = tk.Listbox(sb, height=8, exportselection=False)
        self.stack_list.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=8, pady=(8,4))
        self.stack_list.bind("<<ListboxSelect>>", lambda e: self._show_params_for_selected())
        # ligne 1 : NTSC / Dither / Supprimer / Up / Down
        ttk.Button(sb, text="＋ NTSC", command=lambda: self._add_effect("ntsc")).grid(row=1, column=0, sticky="w", padx=(8,0))
        ttk.Button(sb, text="＋ Dither", command=lambda: self._add_effect("dither")).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Button(sb, text="Supprimer", command=self._remove_selected).grid(row=1, column=2, sticky="w", padx=6)
        ttk.Button(sb, text="↑", width=3, command=lambda: self._move_selected(-1)).grid(row=1, column=3, sticky="e", padx=6)
        ttk.Button(sb, text="↓", width=3, command=lambda: self._move_selected(+1)).grid(row=1, column=4, sticky="e", padx=2)
        # ligne 2 : nouveaux filtres
        ttk.Button(sb, text="＋ Gray", command=lambda: self._add_effect("gray")).grid(row=2, column=0, sticky="w", padx=(8,0), pady=(6,0))
        ttk.Button(sb, text="＋ Sepia", command=lambda: self._add_effect("sepia")).grid(row=2, column=1, sticky="w", padx=6, pady=(6,0))
        ttk.Button(sb, text="＋ Posterize", command=lambda: self._add_effect("posterize")).grid(row=2, column=2, sticky="w", padx=6, pady=(6,0))
        ttk.Button(sb, text="＋ Edge", command=lambda: self._add_effect("edge")).grid(row=2, column=3, sticky="e", padx=6, pady=(6,0))
        ttk.Button(sb, text="＋ Vignette", command=lambda: self._add_effect("vignette")).grid(row=2, column=4, sticky="e", padx=2, pady=(6,0))
        # checkbox pour activer/désactiver l'effet sélectionné (ligne 3)
        self.enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sb, text="Activer l'effet sélectionné", variable=self.enable_var, command=self._toggle_selected).grid(row=3, column=0, columnspan=5, sticky="w", padx=8, pady=(8,8))

        # Paramètres (scroll indépendant + prend toute la largeur disponible)
        cfg = ttk.LabelFrame(left, text="Configure"); cfg.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)
        cfg.rowconfigure(0, weight=1); cfg.columnconfigure(0, weight=1)
        self.params_canvas = tk.Canvas(cfg, highlightthickness=0)
        pscroll = ttk.Scrollbar(cfg, orient="vertical", command=self.params_canvas.yview)
        self.params_container = ttk.Frame(self.params_canvas)
        self.params_container.bind("<Configure>", lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all")))
        self.params_canvas.create_window((0,0), window=self.params_container, anchor="nw")
        self.params_canvas.configure(yscrollcommand=pscroll.set)
        self.params_canvas.grid(row=0, column=0, sticky="nsew")
        pscroll.grid(row=0, column=1, sticky="ns")

        # Accélération / Export options
        opt = ttk.LabelFrame(left, text="Performance & Export"); opt.grid(row=5, column=0, sticky="new", padx=8, pady=(0,8))
        ttk.Checkbutton(opt, text="GPU accel (CUDA) si dispo", variable=self.use_cuda).grid(row=0, column=0, sticky="w", padx=8, pady=(6,2))
        ttk.Checkbutton(opt, text="NVENC (h264_nvenc) si dispo", variable=self.use_nvenc).grid(row=1, column=0, sticky="w", padx=8)
        ttk.Label(opt, text="ffmpeg preset").grid(row=0, column=1, sticky="e", padx=8)
        ttk.Combobox(opt, textvariable=self.ffmpeg_preset, values=["ultrafast","superfast","veryfast","faster","fast","medium","slow"], width=10, state="readonly").grid(row=0, column=2, sticky="w", padx=(0,8))
        ttk.Label(opt, text="threads").grid(row=1, column=1, sticky="e", padx=8)
        ttk.Spinbox(opt, from_=1, to=max(1,(os.cpu_count() or 8)), textvariable=self.ffmpeg_threads, width=6).grid(row=1, column=2, sticky="w")

        # Côté droit : preview + export
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        self.rowconfigure(0, weight=1); self.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(right, bg="#111111")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bottom = ttk.Frame(right)
        bottom.grid(row=1, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Button(bottom, text="Exporter", command=self.on_export).grid(row=0, column=1, padx=(0,12), pady=10, sticky="e")

        # molette
        self.params_canvas.bind_all("<MouseWheel>", lambda e: self.params_canvas.yview_scroll(-1 if e.delta>0 else 1, "units"))

    # ---------- pile ----------
    def _sel_index(self):
        sel = self.stack_list.curselection()
        return None if not sel else sel[0]

    def _add_effect(self, typ):
        name_map = {
            "ntsc": "NTSC", "dither": "Dither",
            "gray": "Gray", "sepia": "Sepia", "posterize": "Posterize",
            "edge": "Edge", "vignette": "Vignette"
        }
        name = name_map.get(typ, typ)
        params = DEFAULT_CFG.copy()
        self.stack.append({"type":typ,"enabled":True,"params":params})
        self.stack_list.insert(tk.END, name)
        self.stack_list.select_clear(0, tk.END); self.stack_list.select_set(tk.END)
        self._show_params_for_selected(); self.request_render()

    def _remove_selected(self):
        idx = self._sel_index()
        if idx is None: return
        self.stack.pop(idx); self.stack_list.delete(idx)
        self._show_params_for_selected(); self.request_render()

    def _move_selected(self, delta):
        idx = self._sel_index()
        if idx is None: return
        new = idx+delta
        if new<0 or new>=len(self.stack): return
        self.stack[idx], self.stack[new] = self.stack[new], self.stack[idx]
        item = self.stack_list.get(idx)
        self.stack_list.delete(idx); self.stack_list.insert(new, item)
        self.stack_list.select_clear(0, tk.END); self.stack_list.select_set(new)
        self.request_render()

    def _toggle_selected(self):
        idx = self._sel_index()
        if idx is None: return
        self.stack[idx]["enabled"]=bool(self.enable_var.get())
        self.request_render()

    # ---------- paramètres ----------
    def _clear_params_ui(self):
        for w in self.params_container.winfo_children(): w.destroy()

    def _slider(self, label, val, setter, vmin, vmax, step=0.01, integer=False):
        row = ttk.Frame(self.params_container); row.pack(fill="x", pady=(6,0))
        ttk.Label(row, text=label).pack(anchor="w")
        var = tk.DoubleVar(value=float(val()))
        s = ttk.Scale(row, from_=vmin, to=vmax, variable=var, orient="horizontal",
                      command=lambda *_: self._on_var_change(var, setter, integer))
        s.pack(fill="x", pady=(2,0))
        lbl = ttk.Label(row, text=f"{int(var.get())}" if integer else f"{var.get():.3f}")
        lbl.pack(anchor="e")
        def wlbl(*_):
            v = var.get()
            if integer:
                v = int(round(v)); var.set(v); lbl.config(text=f"{v}")
            else:
                lbl.config(text=f"{v:.3f}")
        var.trace_add("write", wlbl)
        return var

    def _on_var_change(self, var, setter, integer):
        v = var.get()
        if integer: v = int(round(v)); var.set(v)
        setter(v); self._maybe_render()

    def _show_params_for_selected(self):
        self._clear_params_ui()
        idx = self._sel_index()
        if idx is None:
            self.enable_var.set(True); return
        eff = self.stack[idx]; self.enable_var.set(bool(eff.get("enabled", True)))
        p = eff["params"]

        # sliders communs à NTSC / Dither
        self._slider("ringing", lambda: p.get("ringing", DEFAULT_CFG["ringing"]),
                     lambda v: p.__setitem__("ringing", v), 0, 40, 1, integer=True)
        self._slider("encoding", lambda: p.get("encoding", DEFAULT_CFG["encoding"]),
                     lambda v: p.__setitem__("encoding", int(round(v))), 0, 3, 1, integer=True)
        self._slider("noise", lambda: p.get("noise", DEFAULT_CFG["noise"]),
                     lambda v: p.__setitem__("noise", v), 0.0, 0.03, 0.001)
        self._slider("saturation", lambda: p.get("saturation", DEFAULT_CFG["saturation"]),
                     lambda v: p.__setitem__("saturation", v), 0.0, 3.0, 0.05)
        self._slider("smear", lambda: p.get("smear", DEFAULT_CFG["smear"]),
                     lambda v: p.__setitem__("smear", v), 0.0, 5.0, 0.1)
        self._slider("fringes", lambda: p.get("fringes", DEFAULT_CFG["fringes"]),
                     lambda v: p.__setitem__("fringes", v), 0.0, 3.0, 0.1)

        if eff["type"] == "dither":
            self._slider("pattern type", lambda: p.get("pattern_type", DEFAULT_CFG["pattern_type"]),
                         lambda v: p.__setitem__("pattern_type", int(round(v))), 0, 2, 1, integer=True)
            self._slider("palette type", lambda: p.get("palette_type", DEFAULT_CFG["palette_type"]),
                         lambda v: p.__setitem__("palette_type", int(round(v))), 0, 2, 1, integer=True)
            self._slider("color count", lambda: p.get("color_count", DEFAULT_CFG["color_count"]),
                         lambda v: p.__setitem__("color_count", int(round(v))), 2, 18, 1, integer=True)
            self._slider("distance mode", lambda: p.get("distance_mode", DEFAULT_CFG["distance_mode"]),
                         lambda v: p.__setitem__("distance_mode", int(round(v))), 0, 1, 1, integer=True)
            self._slider("dither strength", lambda: p.get("dither_strength", DEFAULT_CFG["dither_strength"]),
                         lambda v: p.__setitem__("dither_strength", v), 0.0, 3.0, 0.05)
            self._slider("gamma", lambda: p.get("gamma", DEFAULT_CFG["gamma"]),
                         lambda v: p.__setitem__("gamma", v), 0.6, 1.8, 0.01)
            self._slider("pixelStep", lambda: p.get("pixelStep", DEFAULT_CFG["pixelStep"]),
                         lambda v: p.__setitem__("pixelStep", int(round(v))), 1, 16, 1, integer=True)
            self._slider("bayer size (4/8)", lambda: p.get("bayer_size", DEFAULT_CFG["bayer_size"]),
                         lambda v: p.__setitem__("bayer_size", 8 if v >= 6 else 4), 4, 8, 4, integer=True)
            self._slider("color 1", lambda: p.get("color1", DEFAULT_CFG["color1"]),
                         lambda v: p.__setitem__("color1", int(round(v))), 0, 11, 1, integer=True)
            self._slider("color 2", lambda: p.get("color2", DEFAULT_CFG["color2"]),
                         lambda v: p.__setitem__("color2", int(round(v))), 0, 11, 1, integer=True)

        # paramètres pour nouveaux filtres
        if eff["type"] == "gray":
            self._slider("gray strength", lambda: p.get("gray_strength", DEFAULT_CFG["gray_strength"]),
                         lambda v: p.__setitem__("gray_strength", v), 0.0, 1.0, 0.01)
        if eff["type"] == "sepia":
            self._slider("sepia strength", lambda: p.get("sepia_strength", DEFAULT_CFG["sepia_strength"]),
                         lambda v: p.__setitem__("sepia_strength", v), 0.0, 1.0, 0.01)
        if eff["type"] == "posterize":
            self._slider("posterize levels", lambda: p.get("posterize_levels", DEFAULT_CFG["posterize_levels"]),
                         lambda v: p.__setitem__("posterize_levels", int(round(v))), 2, 16, 1, integer=True)
        if eff["type"] == "edge":
            self._slider("edge strength", lambda: p.get("edge_strength", DEFAULT_CFG["edge_strength"]),
                         lambda v: p.__setitem__("edge_strength", v), 0.0, 1.0, 0.01)
        if eff["type"] == "vignette":
            self._slider("vignette strength", lambda: p.get("vignette_strength", DEFAULT_CFG["vignette_strength"]),
                         lambda v: p.__setitem__("vignette_strength", v), 0.0, 1.0, 0.01)
            self._slider("vignette radius", lambda: p.get("vignette_radius", DEFAULT_CFG["vignette_radius"]),
                         lambda v: p.__setitem__("vignette_radius", v), 0.2, 1.5, 0.01)

    # ---------- worker de rendu (asynchrone) ----------
    def _render_worker(self):
        while True:
            task = self._render_q.get()
            if task is None:
                continue
            try:
                kind, payload = task
                if kind == "image":
                    src, stack = payload
                    arr = pil_to_np(src)
                    out = arr
                    for e in stack:
                        out = apply_effect(out, e)
                    result = np_to_pil(out, keep_alpha=has_alpha(src))
                else:  # frame
                    frame, stack = payload
                    out = apply_stack_to_frame(frame, stack)
                    result = Image.fromarray(out)
                self._post_preview(result)
            except Exception as e:
                print("[Render worker]", e)

    def _post_preview(self, pil_img):
        # transférer sur le thread UI
        self.after(0, lambda: self._show_on_canvas(pil_img))

    # ---------- fichier / vidéo ----------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier",
            filetypes=[("Media","*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.mov;*.mkv;*.avi"),
                       ("Images","*.png;*.jpg;*.jpeg;*.webp"),
                       ("Vidéos","*.mp4;*.mov;*.mkv;*.avi")]
        )
        if not path: return
        self.load_input(path)

    def _rebuild_video_slider(self, duration):
        for w in self.video_box.winfo_children(): w.destroy()
        self.video_box.columnconfigure(0, weight=1)
        ttk.Label(self.video_box, text=f"Position (0..{duration:.2f} s)").grid(row=0, column=0, sticky="w", padx=8, pady=(8,0))
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub = ttk.Scale(self.video_box, from_=0, to=max(1.0,duration),
                               variable=self.scrub_var, orient="horizontal",
                               command=lambda *_: self._maybe_render())
        self.scrub.grid(row=1, column=0, sticky="ew", padx=8, pady=(2,8))

    def load_input(self, path):
        global USE_CUDA_DEFAULT
        USE_CUDA_DEFAULT = self.use_cuda.get() and cuda_available()

        if self.video_clip is not None:
            try: self.video_clip.close()
            except: pass
            self.video_clip=None

        self.input_path=path
        ext = os.path.splitext(path)[1].lower()
        self.input_is_video = ext in [".mp4",".mov",".mkv",".avi"]

        try:
            if self.input_is_video:
                self.video_clip = VideoFileClip(path)
                self.video_duration = float(self.video_clip.duration or 0)
                self.info_label.config(text=f"Vidéo: {os.path.basename(path)} • {self.video_duration:.2f}s • {self.video_clip.w}x{self.video_clip.h}")
                self._rebuild_video_slider(self.video_duration)
                self.video_box.grid()  # show
            else:
                self.loaded_image = imread_any(path)
                self.info_label.config(text=f"Image: {os.path.basename(path)} • {self.loaded_image.width}x{self.loaded_image.height}")
                self.video_box.grid_remove()
        except Exception as e:
            messagebox.showerror("Erreur d'ouverture", str(e)); return
        self.request_render()

    # ---------- rendu ----------
    def _maybe_render(self):
        if self.live_preview.get(): self.request_render()

    def request_render(self):
        # annule l'éventuelle demande précédente et garde la dernière
        while not self._render_q.empty():
            try: self._render_q.get_nowait()
            except queue.Empty: break
        if not self.input_path: return
        if self.input_is_video:
            t=max(0.0, min(self.video_duration, float(self.scrub_var.get())))
            frame = self.video_clip.get_frame(t).astype(np.uint8)
            frame = self._fit_preview_np(frame)
            self._render_q.put(("frame", (frame, list(self.stack))))
        else:
            img = self._fit_preview_pil(self.loaded_image.copy())
            self._render_q.put(("image", (img, list(self.stack))))

    def _fit_preview_pil(self, img, max_ratio=0.92):
        # on redimensionne via Pillow (qualité Lanczos) pour les images fixes
        cw = int(self.canvas.winfo_width() or 960); ch = int(self.canvas.winfo_height() or 540)
        maxw = int(cw*max_ratio); maxh=int(ch*max_ratio)
        w,h = img.size; s = min(maxw/max(1,w), maxh/max(1,h), 1.0)
        if s<1.0: img = img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.LANCZOS)
        return img

    def _fit_preview_np(self, frame_np, max_pixels=1280*720):
        # OpenCV (plus rapide) pour les frames vidéo
        h,w = frame_np.shape[:2]
        if h*w<=max_pixels: return frame_np
        s = math.sqrt(max_pixels/(h*w))
        nh,nw = max(1,int(h*s)), max(1,int(w*s))
        return cv2.resize(frame_np,(nw,nh), interpolation=cv2.INTER_AREA)

    def _show_on_canvas(self, pil_img):
        self.canvas.delete("all")
        self.preview_imgtk = ImageTk.PhotoImage(pil_img)
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        self.canvas.create_image(cw//2, ch//2, image=self.preview_imgtk, anchor="center")

    # ---------- export (unique) ----------
    def on_export(self):
        if not self.input_path:
            messagebox.showinfo("Info", "Charge d'abord un fichier image/vidéo."); return
        if self.input_is_video:
            out = filedialog.asksaveasfilename(title="Exporter la vidéo", defaultextension=".mp4",
                                               filetypes=[("MP4","*.mp4"),("MOV","*.mov"),("MKV","*.mkv"),("AVI","*.avi")])
            if not out: return
            try:
                final = self._process_video_full(self.input_path, out, stack=self.stack)
                messagebox.showinfo("Succès", f"Vidéo exportée :\n{final}")
            except Exception as e:
                messagebox.showerror("Erreur export vidéo", str(e))
        else:
            out = filedialog.asksaveasfilename(title="Exporter l'image", defaultextension=".png",
                                               filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("WEBP","*.webp")])
            if not out: return
            try:
                src = imread_any(self.input_path); img = apply_stack_to_pil(src, self.stack)
                img.save(out); messagebox.showinfo("Succès", f"Image exportée :\n{out}")
            except Exception as e:
                messagebox.showerror("Erreur export image", str(e))

    # export vidéo avec options NVENC / threads / preset
    def _process_video_full(self, input_path, output_path, stack, bitrate="6M"):
        global USE_CUDA_DEFAULT
        USE_CUDA_DEFAULT = self.use_cuda.get() and cuda_available()

        clip = VideoFileClip(input_path)
        def frame_fn(frame):
            return apply_stack_to_frame(frame.astype(np.uint8), stack)
        out_clip = apply_framewise(clip, frame_fn).set_audio(clip.audio)

        ext = os.path.splitext(output_path)[1].lower()
        if ext not in [".mp4", ".mov", ".mkv", ".avi"]:
            output_path += ".mp4"

        codec = "h264_nvenc" if self.use_nvenc.get() else "libx264"
        out_clip.write_videofile(
            output_path,
            codec=codec,
            audio_codec="aac",
            bitrate=bitrate,
            fps=clip.fps or 25,
            threads=int(self.ffmpeg_threads.get()),
            preset=self.ffmpeg_preset.get(),
            verbose=False,
            logger=None
        )
        clip.close(); out_clip.close()
        return output_path

    # ---------- config save / load ----------
    def _make_serializable(self, obj):
        # convertit récursivement numpy scalaires / types non JSON en types Python natifs
        if isinstance(obj, dict):
            return {self._make_serializable(k): self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # bool, int, float, str, None restent inchangés
        return obj

    def on_save_config(self):
        # propose un fichier .FXOption et sauvegarde le chemin d'image + pile + options
        out = filedialog.asksaveasfilename(title="Sauvegarder la configuration", defaultextension=".FXOption",
                                           filetypes=[("FXOption","*.FXOption"),("JSON","*.json")])
        if not out:
            return
        cfg = {
            "input_path": self.input_path,
            "stack": copy.deepcopy(self.stack),
            "use_cuda": bool(self.use_cuda.get()),
            "use_nvenc": bool(self.use_nvenc.get()),
            "ffmpeg_preset": str(self.ffmpeg_preset.get()),
            "ffmpeg_threads": int(self.ffmpeg_threads.get()),
            "live_preview": bool(self.live_preview.get()),
        }
        try:
            serial = self._make_serializable(cfg)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Succès", f"Configuration sauvegardée :\n{out}")
        except Exception as e:
            messagebox.showerror("Erreur sauvegarde", str(e))

    def on_load_config(self):
        path = filedialog.askopenfilename(title="Ouvrir une configuration", filetypes=[("FXOption/JSON","*.FXOption;*.json"),("JSON","*.json"),("FXOption","*.FXOption")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_loaded_config(cfg)
            messagebox.showinfo("Succès", f"Configuration chargée :\n{path}")
        except Exception as e:
            messagebox.showerror("Erreur ouverture config", str(e))

    def _apply_loaded_config(self, cfg):
        # Applique les valeurs depuis un dict (robuste aux champs manquants)
        ip = cfg.get("input_path")
        if ip and os.path.exists(ip):
            try:
                self.load_input(ip)
            except Exception:
                # ne pas stopper si l'image pose problème, on continue à charger la pile
                self.input_path = None; self.loaded_image = None
                self.info_label.config(text=f"Fichier référencé introuvable : {os.path.basename(ip)}")
        else:
            # si chemin non fourni ou introuvable, on vide l'entrée
            self.input_path = None; self.loaded_image = None
            if ip:
                self.info_label.config(text=f"Fichier référencé introuvable : {os.path.basename(ip)}")
            else:
                self.info_label.config(text="Aucun fichier chargé")

        # options générales
        try: self.use_cuda.set(bool(cfg.get("use_cuda", self.use_cuda.get())))
        except: pass
        try: self.use_nvenc.set(bool(cfg.get("use_nvenc", self.use_nvenc.get())))
        except: pass
        try: self.ffmpeg_preset.set(str(cfg.get("ffmpeg_preset", self.ffmpeg_preset.get())))
        except: pass
        try: self.ffmpeg_threads.set(int(cfg.get("ffmpeg_threads", self.ffmpeg_threads.get())))
        except: pass
        try: self.live_preview.set(bool(cfg.get("live_preview", self.live_preview.get())))
        except: pass

        # pile d'effets
        loaded_stack = cfg.get("stack", [])
        # validation minimale / normalisation
        new_stack = []
        allowed = ("ntsc", "dither", "gray", "sepia", "posterize", "edge", "vignette")
        for e in loaded_stack:
            if not isinstance(e, dict): continue
            typ = e.get("type")
            if typ not in allowed: continue
            params = e.get("params", {})
            if not isinstance(params, dict):
                params = {}
            # merge avec DEFAULT_CFG pour garantir tous les champs
            merged = DEFAULT_CFG.copy()
            merged.update(params)
            new_stack.append({"type": typ, "enabled": bool(e.get("enabled", True)), "params": merged})
        self.stack = new_stack

        # rebuild listbox UI
        self.stack_list.delete(0, tk.END)
        name_map = {"ntsc":"NTSC","dither":"Dither","gray":"Gray","sepia":"Sepia","posterize":"Posterize","edge":"Edge","vignette":"Vignette"}
        for e in self.stack:
            self.stack_list.insert(tk.END, name_map.get(e["type"], e["type"]))
        # select first item if any
        if len(self.stack) > 0:
            self.stack_list.select_clear(0, tk.END); self.stack_list.select_set(0)
        self._show_params_for_selected()
        self.request_render()

def main():
    app = EffectsGUI()
    # Par défaut : Dither puis NTSC (comme tes captures de départ)
    app.stack = [
        {"type":"dither","enabled":True,"params":DEFAULT_CFG.copy()},
        {"type":"ntsc","enabled":True,"params":DEFAULT_CFG.copy()},
    ]
    app.stack_list.insert(tk.END, "Dither"); app.stack_list.insert(tk.END, "NTSC")
    app.request_render(); app.mainloop()

if __name__ == "__main__":
    main()
