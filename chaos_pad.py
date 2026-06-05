"""
chaos_pad.py  —  Chaos Pad dla ED_Waves
========================================
8-kątny pad XY  +  oś Z (suwak głębokości)
Tryby: ręczny / auto-oscylacja (random walk + Lissajous)
Steruje: harmoniki, ADSR, f0, polifonia, ambience

Uruchomienie standalone:
    python chaos_pad.py

Integracja z ED_Waves v3 — importuj ChaosPadWindow i wywołaj:
    pad = ChaosPadWindow(callback=twoja_funkcja)
    pad.show()
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo
import numpy as np
import math, random, os, tempfile

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

try:
    import scipy.io.wavfile as wav
    from scipy.signal import butter, sosfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ═══════════════════════════════════════════════════════════════════
#  8 stref pada  (kąt, nazwa, kolor, profil dźwięczności)
# ═══════════════════════════════════════════════════════════════════
# Strefy: centrum (0) + 8 rogów (1-8) rozmieszczonych co 45°
# Każda strefa ma bazowy "klimat" — blending interpoluje między nimi

ZONES = [
    # idx  nazwa            kolor-RGB          opis klimatu
    (0,  "NEUTRAL",       (0.40, 0.40, 0.45), "czysta sinusoida, f0=440, bez harmonik"),
    (1,  "WARM PAD",      (0.85, 0.45, 0.10), "pad: wolny atak, bogate harmoniki niskie"),
    (2,  "BRIGHT LEAD",   (0.20, 0.75, 0.95), "lead: szybki atak, górne harmoniki"),
    (3,  "DEEP BASS",     (0.55, 0.10, 0.85), "bass: f0 sub, mocne 2H+3H"),
    (4,  "PLUCK",         (0.10, 0.85, 0.35), "pluck: impuls, szybki decay do 0"),
    (5,  "ORGAN",         (0.95, 0.90, 0.15), "organ: równe harmoniki 1-5"),
    (6,  "CHOIR",         (0.90, 0.30, 0.60), "choir: formant, polifonia, vibrato"),
    (7,  "NOISE AMBIENT", (0.30, 0.55, 0.90), "ambient: szum + długi reverb"),
    (8,  "BELL",          (0.95, 0.65, 0.20), "bell: FM, inharmoniczne, krótki decay"),
]

# Parametry bazowe dla każdej strefy:
# [f0_hz, h1,h2,h3,h4,h5, attack,decay,sustain,release, poly, ambience, noise]
ZONE_PARAMS = {
    "NEUTRAL":       dict(f0=440, h=[1.0,0.0,0.0,0.0,0.0], adsr=[0.01,0.1,0.8,0.2], poly=1, amb=0.0, noise=0.0),
    "WARM PAD":      dict(f0=220, h=[1.0,0.6,0.3,0.15,0.07],adsr=[0.35,0.15,0.9,0.6], poly=3, amb=0.4, noise=0.01),
    "BRIGHT LEAD":   dict(f0=440, h=[1.0,0.7,0.5,0.3,0.15], adsr=[0.01,0.08,0.7,0.1], poly=1, amb=0.1, noise=0.0),
    "DEEP BASS":     dict(f0=55,  h=[1.0,0.8,0.4,0.1,0.05], adsr=[0.01,0.15,0.5,0.12],poly=1, amb=0.0, noise=0.0),
    "PLUCK":         dict(f0=330, h=[1.0,0.5,0.2,0.05,0.0], adsr=[0.002,0.3,0.0,0.05],poly=2, amb=0.2, noise=0.0),
    "ORGAN":         dict(f0=264, h=[1.0,1.0,1.0,0.7,0.4],  adsr=[0.001,0.0,1.0,0.001],poly=4,amb=0.05,noise=0.0),
    "CHOIR":         dict(f0=220, h=[1.0,0.6,0.4,0.2,0.1],  adsr=[0.20,0.1,0.8,0.3],  poly=5, amb=0.5, noise=0.03),
    "NOISE AMBIENT": dict(f0=110, h=[0.3,0.2,0.1,0.05,0.02],adsr=[0.5,0.3,0.7,0.8],   poly=2, amb=0.9, noise=0.7),
    "BELL":          dict(f0=440, h=[1.0,0.3,0.6,0.1,0.4],  adsr=[0.001,0.6,0.0,0.5], poly=1, amb=0.3, noise=0.0),
}

# Kąty stref 1-8 (0° = góra, zgodnie z ruchem wskazówek)
ZONE_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # strefy 1-8


# ═══════════════════════════════════════════════════════════════════
#  Interpolacja parametrów z pozycji XYZ pada
# ═══════════════════════════════════════════════════════════════════
def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_params(p1, p2, t):
    """Interpoluje dwa słowniki parametrów."""
    out = {}
    for k in p1:
        v1, v2 = p1[k], p2[k]
        if isinstance(v1, list):
            out[k] = [_lerp(a, b, t) for a, b in zip(v1, v2)]
        else:
            out[k] = _lerp(v1, v2, t)
    return out

def xy_to_params(nx, ny, z=0.5):
    """
    nx, ny ∈ [-1..+1]  (centrum = 0,0)
    z      ∈ [0..1]    (głębokość — suwak)

    Zwraca słownik parametrów brzmienia.
    """
    r = math.sqrt(nx*nx + ny*ny)  # 0=centrum, 1=krawędź
    r = min(r, 1.0)

    # kąt od środka (0° = góra)
    ang_deg = (math.degrees(math.atan2(nx, -ny)) + 360) % 360

    # znajdź dwie sąsiednie strefy i ich wagi
    sector = ang_deg / 45.0        # 0..8
    idx_lo = int(sector) % 8       # strefa lewa
    idx_hi = (idx_lo + 1) % 8     # strefa prawa
    t_ang  = sector - int(sector)  # 0..1 między strefami

    name_lo = ZONES[idx_lo + 1][1]
    name_hi = ZONES[idx_hi + 1][1]
    p_lo    = ZONE_PARAMS[name_lo]
    p_hi    = ZONE_PARAMS[name_hi]
    p_edge  = _lerp_params(p_lo, p_hi, t_ang)

    # blend z centrum wg r
    p_center = ZONE_PARAMS["NEUTRAL"]
    p_blend  = _lerp_params(p_center, p_edge, r)

    # oś Z: głębokość = wzmacnia harmoniki + ambience
    p_blend['h']   = [min(1.0, v * (0.5 + z)) for v in p_blend['h']]
    p_blend['amb'] = min(1.0, p_blend['amb'] * (0.3 + z * 1.4))
    p_blend['noise'] = min(1.0, p_blend['noise'] * (0.2 + z * 1.6))

    # f0 zaokrąglij do skali równomiernie temperowanej (A=440)
    raw_f0 = p_blend['f0']
    midi   = 69 + 12 * math.log2(max(raw_f0, 20) / 440)
    midi_r = round(midi)
    p_blend['f0'] = 440 * (2 ** ((midi_r - 69) / 12))
    p_blend['midi_note'] = midi_r

    return p_blend


# ═══════════════════════════════════════════════════════════════════
#  Synteza z parametrów pada
# ═══════════════════════════════════════════════════════════════════
SR = 44100

def _adsr_env(n, a, d, s, r):
    ia = min(int(a * SR), n)
    id_ = min(int(d * SR), n - ia)
    ir = min(int(r * SR), n)
    isu = max(0, n - ia - id_ - ir)
    parts = []
    if ia:  parts.append(np.linspace(0, 1, ia))
    if id_: parts.append(np.linspace(1, s, id_))
    if isu: parts.append(np.full(isu, s))
    if ir:  parts.append(np.linspace(s, 0, ir))
    env = np.concatenate(parts) if parts else np.ones(n)
    if len(env) < n: env = np.pad(env, (0, n - len(env)))
    return env[:n]

def _reverb_simple(wave, wet=0.3, sr=SR):
    """Prosty reverb przez splatanie z exp-decay IR."""
    if wet < 0.01: return wave
    ir_len = int(sr * 0.8 * wet)
    if ir_len < 4: return wave
    ir = np.exp(-np.linspace(0, 8, ir_len)) * np.random.normal(0, 1, ir_len)
    ir /= np.max(np.abs(ir) + 1e-9)
    conv = np.convolve(wave, ir)[:len(wave)]
    return wave * (1 - wet) + conv * wet

def synthesize(params, duration=1.0):
    """Generuje falę z parametrów pada."""
    n    = int(SR * duration)
    t    = np.linspace(0, duration, n, endpoint=False)
    f0   = max(20.0, float(params['f0']))
    h    = params['h']
    poly = max(1, int(round(params['poly'])))
    adsr = params['adsr']
    amb  = float(params['amb'])
    noise_lv = float(params['noise'])

    # addytywna synteza harmonik
    wave = np.zeros(n)
    for i, w in enumerate(h):
        if w < 1e-4: continue
        freq_i = f0 * (i + 1)
        if freq_i > 20000: break
        wave += w * np.sin(2 * np.pi * freq_i * t)

    # polifonia — dodaj kopie na interwałach kwinty i tercji
    if poly > 1:
        intervals = [7, 4, 12, 3, 9]  # semitony
        for k in range(min(poly - 1, len(intervals))):
            fr = f0 * (2 ** (intervals[k] / 12))
            ch_wave = np.zeros(n)
            for i, w in enumerate(h):
                if w < 1e-4: continue
                fi = fr * (i + 1)
                if fi > 20000: break
                ch_wave += w * 0.6 * np.sin(2 * np.pi * fi * t)
            wave += ch_wave

    # szum
    if noise_lv > 0.01:
        wave += np.random.normal(0, noise_lv, n)

    # ADSR
    a_t, d_t, s_lv, r_t = adsr
    env = _adsr_env(n, a_t, d_t, s_lv, r_t)
    wave *= env

    # normalizacja
    mx = np.max(np.abs(wave))
    if mx > 1e-9: wave /= mx

    # reverb / ambience
    if SCIPY_OK and amb > 0.01:
        wave = _reverb_simple(wave, wet=amb * 0.7)
        mx2 = np.max(np.abs(wave))
        if mx2 > 1e-9: wave /= mx2

    return wave


# ═══════════════════════════════════════════════════════════════════
#  Auto-oscylator (random walk + Lissajous)
# ═══════════════════════════════════════════════════════════════════
class AutoOscillator:
    """Generuje płynną trajektorię po padzie."""

    LISSAJOUS = "lissajous"
    RANDOM_WALK = "random_walk"
    SPIRAL = "spiral"

    def __init__(self, mode=RANDOM_WALK, speed=0.008):
        self.mode  = mode
        self.speed = speed
        self.x = 0.0; self.y = 0.0
        self.phase_a = random.uniform(0, math.pi*2)
        self.phase_b = random.uniform(0, math.pi*2)
        self.freq_a  = random.uniform(0.7, 1.5)
        self.freq_b  = random.uniform(0.9, 2.0)
        self.vx = random.uniform(-0.012, 0.012)
        self.vy = random.uniform(-0.012, 0.012)
        self.spiral_r = 0.0
        self.spiral_dir = 1
        self.t = 0.0

    def step(self):
        self.t += self.speed

        if self.mode == self.LISSAJOUS:
            a = self.freq_a; b = self.freq_b
            da = random.uniform(-0.0003, 0.0003)
            db = random.uniform(-0.0003, 0.0003)
            self.freq_a = max(0.5, min(3.0, a + da))
            self.freq_b = max(0.5, min(3.0, b + db))
            self.x = math.sin(self.freq_a * self.t + self.phase_a)
            self.y = math.sin(self.freq_b * self.t + self.phase_b)

        elif self.mode == self.RANDOM_WALK:
            self.vx += random.uniform(-0.001, 0.001)
            self.vy += random.uniform(-0.001, 0.001)
            self.vx = max(-0.025, min(0.025, self.vx))
            self.vy = max(-0.025, min(0.025, self.vy))
            self.x += self.vx
            self.y += self.vy
            # odbicie od krawędzi
            r = math.sqrt(self.x*self.x + self.y*self.y)
            if r > 0.95:
                self.x *= 0.9; self.y *= 0.9
                self.vx *= -0.5; self.vy *= -0.5

        elif self.mode == self.SPIRAL:
            self.spiral_r += self.spiral_dir * 0.003
            if self.spiral_r > 0.95: self.spiral_dir = -1
            if self.spiral_r < 0.05: self.spiral_dir = 1
            self.x = self.spiral_r * math.sin(self.t * 1.5)
            self.y = self.spiral_r * math.cos(self.t * 1.0)

        return self.x, self.y

    def teleport(self, x, y):
        self.x = x; self.y = y
        self.vx = 0; self.vy = 0


# ═══════════════════════════════════════════════════════════════════
#  Widget pada (DrawingArea)
# ═══════════════════════════════════════════════════════════════════
class ChaosPadWidget(Gtk.DrawingArea):
    def __init__(self, on_change=None):
        super().__init__()
        self.on_change = on_change   # callback(nx, ny, z)
        self.nx = 0.0; self.ny = 0.0
        self.z  = 0.5
        self.dragging = False

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect("draw",                  self._draw)
        self.connect("button-press-event",    self._press)
        self.connect("button-release-event",  self._release)
        self.connect("motion-notify-event",   self._motion)

        # historia ścieżki
        self.trail = []
        self.MAX_TRAIL = 80

    def set_pos(self, nx, ny):
        self.nx = max(-1.0, min(1.0, nx))
        self.ny = max(-1.0, min(1.0, ny))
        self.trail.append((self.nx, self.ny))
        if len(self.trail) > self.MAX_TRAIL:
            self.trail.pop(0)
        self.queue_draw()

    # ── rysowanie ────────────────────────────────────────────────
    def _draw(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        cx = w / 2; cy = h / 2
        R  = min(cx, cy) - 10   # promień pada

        # tło
        cr.set_source_rgb(0.07, 0.07, 0.10); cr.paint()

        # okrąg pada
        cr.set_source_rgba(0.25, 0.25, 0.30, 0.8)
        cr.arc(cx, cy, R, 0, 2*math.pi); cr.fill()

        # osie
        cr.set_source_rgba(0.35, 0.35, 0.40, 0.5)
        cr.set_line_width(1)
        cr.move_to(cx - R, cy); cr.line_to(cx + R, cy); cr.stroke()
        cr.move_to(cx, cy - R); cr.line_to(cx, cy + R); cr.stroke()

        # 8 stref — sektory
        for i in range(8):
            ang_start = math.radians(i * 45 - 22.5 - 90)
            ang_end   = math.radians((i+1) * 45 - 22.5 - 90)
            _, name, col, _ = ZONES[i + 1]
            r, g, b = col

            # wycinek strefy (tylko obwódka + label)
            cr.set_source_rgba(r, g, b, 0.18)
            cr.move_to(cx, cy)
            cr.arc(cx, cy, R, ang_start, ang_end)
            cr.close_path(); cr.fill()

            # linia podziału
            cr.set_source_rgba(r, g, b, 0.4)
            cr.set_line_width(0.8)
            mid_ang = (ang_start + ang_end) / 2
            cr.move_to(cx, cy)
            cr.line_to(cx + R * math.cos(mid_ang),
                       cy + R * math.sin(mid_ang))
            cr.stroke()

            # etykieta strefy
            lx = cx + (R * 0.72) * math.cos(mid_ang)
            ly = cy + (R * 0.72) * math.sin(mid_ang)
            cr.set_source_rgba(r, g, b, 0.95)
            cr.set_font_size(8.5)
            te = cr.text_extents(name)
            cr.move_to(lx - te.width/2, ly + te.height/2)
            cr.show_text(name)

        # siatka kół
        for frac in [0.33, 0.66, 1.0]:
            cr.set_source_rgba(0.40, 0.40, 0.48, 0.25)
            cr.set_line_width(0.6)
            cr.arc(cx, cy, R * frac, 0, 2*math.pi); cr.stroke()

        # ślad ruchu
        if len(self.trail) > 1:
            for k in range(len(self.trail) - 1):
                alpha = (k / len(self.trail)) * 0.6
                tx0, ty0 = self.trail[k]
                tx1, ty1 = self.trail[k+1]
                px0 = cx + tx0 * R; py0 = cy - ty0 * R
                px1 = cx + tx1 * R; py1 = cy - ty1 * R
                # kolor śladu zależy od strefy
                ang = (math.degrees(math.atan2(tx0, -ty0)) + 360) % 360
                zi  = int(ang / 45) % 8
                rc, gc, bc = ZONES[zi + 1][2]
                cr.set_source_rgba(rc, gc, bc, alpha)
                cr.set_line_width(1.5)
                cr.move_to(px0, py0); cr.line_to(px1, py1); cr.stroke()

        # kursor (duże kółko z kolorem strefy)
        px = cx + self.nx * R
        py = cy - self.ny * R
        ang = (math.degrees(math.atan2(self.nx, -self.ny)) + 360) % 360
        zi  = int(ang / 45) % 8
        rc, gc, bc = ZONES[zi + 1][2]
        r_cur = 0.0 + math.sqrt(self.nx*self.nx + self.ny*self.ny)

        cr.set_source_rgba(rc, gc, bc, 0.9)
        cr.arc(px, py, 10 + r_cur * 4, 0, 2*math.pi); cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.arc(px, py, 10 + r_cur * 4, 0, 2*math.pi)
        cr.set_line_width(1.5); cr.stroke()

        # crosshair na kursorze
        cr.set_source_rgba(1, 1, 1, 0.5); cr.set_line_width(1)
        cr.move_to(px-14, py); cr.line_to(px+14, py); cr.stroke()
        cr.move_to(px, py-14); cr.line_to(px, py+14); cr.stroke()

        # centrum mark
        cr.set_source_rgba(0.6, 0.6, 0.7, 0.7)
        cr.arc(cx, cy, 4, 0, 2*math.pi); cr.fill()

        # etykieta pozycji
        cr.set_source_rgba(0.8, 0.8, 0.9, 0.85)
        cr.set_font_size(9)
        cr.move_to(6, h - 6)
        cr.show_text(f"X={self.nx:+.2f}  Y={self.ny:+.2f}  Z={self.z:.2f}")

    # ── mysz ─────────────────────────────────────────────────────
    def _screen_to_norm(self, ex, ey):
        w = self.get_allocated_width()
        h = self.get_allocated_height()
        cx = w/2; cy = h/2; R = min(cx, cy) - 10
        nx = (ex - cx) / R
        ny = -(ey - cy) / R
        r = math.sqrt(nx*nx + ny*ny)
        if r > 1.0: nx /= r; ny /= r
        return nx, ny

    def _press(self, widget, event):
        if event.button == 1:
            self.dragging = True
            nx, ny = self._screen_to_norm(event.x, event.y)
            self.set_pos(nx, ny)
            if self.on_change: self.on_change(self.nx, self.ny, self.z)

    def _release(self, widget, event):
        if event.button == 1:
            self.dragging = False

    def _motion(self, widget, event):
        if self.dragging:
            nx, ny = self._screen_to_norm(event.x, event.y)
            self.set_pos(nx, ny)
            if self.on_change: self.on_change(self.nx, self.ny, self.z)


# ═══════════════════════════════════════════════════════════════════
#  Główne okno Chaos Pada
# ═══════════════════════════════════════════════════════════════════
class ChaosPadWindow(Gtk.Window):
    """
    Standalone lub jako sub-okno ED_Waves.
    callback(wave: np.ndarray, params: dict) — wywoływany przy każdej zmianie.
    """

    TICK_MS = 40   # ms między krokami auto-oscylatora (~25 fps)

    def __init__(self, callback=None):
        Gtk.Window.__init__(self, title="Chaos Pad — ED_Waves")
        self.set_default_size(780, 620)
        self.connect("destroy", self._on_destroy)

        self.wave_callback = callback  # fn(wave, params) lub None

        self.auto_mode  = False
        self.osc        = AutoOscillator(AutoOscillator.RANDOM_WALK, speed=0.008)
        self.current_params = ZONE_PARAMS["NEUTRAL"].copy()
        self.current_wave   = np.zeros(44100)
        self.duration   = 1.0
        self._timer_id  = None

        self._build_ui()
        self._update_info()

    # ══════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        root.set_border_width(4)
        self.add(root)

        # ── pasek górny: tryb auto + oscylator ───────────────────
        tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.pack_start(tb, False, False, 0)

        self.auto_btn = Gtk.ToggleButton(label="⟳ Auto OFF")
        self.auto_btn.connect("toggled", self._toggle_auto)
        tb.pack_start(self.auto_btn, False, False, 0)

        tb.pack_start(Gtk.Label(label="Tryb:"), False, False, 0)
        self.osc_combo = Gtk.ComboBoxText()
        for m in ["random_walk", "lissajous", "spiral"]:
            self.osc_combo.append_text(m)
        self.osc_combo.set_active(0)
        self.osc_combo.connect("changed", self._osc_mode_changed)
        tb.pack_start(self.osc_combo, False, False, 0)

        tb.pack_start(Gtk.Label(label="Szybkość:"), False, False, 0)
        self.speed_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.speed_sc.set_range(0.001, 0.05); self.speed_sc.set_value(0.008)
        self.speed_sc.set_digits(3); self.speed_sc.set_size_request(120, -1)
        self.speed_sc.connect("value-changed",
                              lambda w: setattr(self.osc, 'speed', w.get_value()))
        tb.pack_start(self.speed_sc, False, False, 0)

        tb.pack_start(Gtk.Label(label="  Czas[s]:"), False, False, 0)
        self.dur_spin = Gtk.SpinButton()
        self.dur_spin.set_range(0.1, 5.0); self.dur_spin.set_value(1.0)
        self.dur_spin.set_increments(0.1, 0.5); self.dur_spin.set_digits(1)
        self.dur_spin.connect("value-changed",
                              lambda w: setattr(self, 'duration', w.get_value()))
        tb.pack_start(self.dur_spin, False, False, 0)

        for lbl, cb in [("▶ Play", self._play), ("⏹ Stop", self._stop),
                        ("💾 WAV",  self._save_wav)]:
            b = Gtk.Button(label=lbl); b.connect("clicked", cb)
            tb.pack_start(b, False, False, 0)

        # ── środek: pad + boczny panel ───────────────────────────
        mid = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        root.pack_start(mid, True, True, 0)

        # pad
        self.pad = ChaosPadWidget(on_change=self._pad_changed)
        self.pad.set_size_request(460, 460)
        mid.pack_start(self.pad, True, True, 0)

        # prawy panel: suwak Z + info + podgląd fali
        rp = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        rp.set_border_width(4); mid.pack_start(rp, False, False, 0)

        rp.pack_start(Gtk.Label(label="Oś Z (głębokość):"), False, False, 0)
        self.z_scale = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
        self.z_scale.set_range(0.0, 1.0); self.z_scale.set_value(0.5)
        self.z_scale.set_inverted(True)   # góra = 1.0
        self.z_scale.set_digits(2); self.z_scale.set_size_request(40, 200)
        self.z_scale.connect("value-changed", self._z_changed)
        rp.pack_start(self.z_scale, False, False, 0)

        # separator
        rp.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        # info parametrów
        self.info_label = Gtk.Label(label="")
        self.info_label.set_xalign(0)
        self.info_label.set_line_wrap(True)
        self.info_label.set_width_chars(26)
        rp.pack_start(self.info_label, False, False, 0)

        rp.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        # podgląd fali (mini DrawingArea)
        rp.pack_start(Gtk.Label(label="Podgląd fali:"), False, False, 0)
        self.wave_preview = Gtk.DrawingArea()
        self.wave_preview.set_size_request(220, 100)
        self.wave_preview.connect("draw", self._draw_wave_preview)
        rp.pack_start(self.wave_preview, False, False, 0)

        # ── pasek statusu ─────────────────────────────────────────
        self.status = Gtk.Label(label="Gotowy — kliknij pad lub włącz Auto")
        self.status.set_xalign(0)
        root.pack_start(self.status, False, False, 0)

    # ══════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════
    def _pad_changed(self, nx, ny, z):
        self.current_params = xy_to_params(nx, ny, z)
        self.current_wave   = synthesize(self.current_params, self.duration)
        self._update_info()
        self.wave_preview.queue_draw()
        if self.wave_callback:
            self.wave_callback(self.current_wave.copy(), self.current_params.copy())

    def _z_changed(self, widget):
        self.pad.z = widget.get_value()
        self._pad_changed(self.pad.nx, self.pad.ny, self.pad.z)

    def _toggle_auto(self, btn):
        self.auto_mode = btn.get_active()
        btn.set_label("⟳ Auto ON" if self.auto_mode else "⟳ Auto OFF")
        if self.auto_mode:
            self._timer_id = GLib.timeout_add(self.TICK_MS, self._auto_tick)
        else:
            if self._timer_id:
                GLib.source_remove(self._timer_id)
                self._timer_id = None

    def _auto_tick(self):
        if not self.auto_mode:
            return False
        nx, ny = self.osc.step()
        self.pad.set_pos(nx, ny)
        self._pad_changed(nx, ny, self.pad.z)
        return True  # kontynuuj

    def _osc_mode_changed(self, combo):
        modes = {"random_walk": AutoOscillator.RANDOM_WALK,
                 "lissajous":   AutoOscillator.LISSAJOUS,
                 "spiral":      AutoOscillator.SPIRAL}
        m = modes.get(combo.get_active_text(), AutoOscillator.RANDOM_WALK)
        self.osc.mode = m

    def _play(self, widget=None):
        if not PYGAME_OK: return
        wave = self.current_wave
        if len(wave) < 2: return
        pygame.mixer.music.load(self._tmp(wave)); pygame.mixer.music.play()

    def _stop(self, widget=None):
        if PYGAME_OK: pygame.mixer.music.stop()

    def _save_wav(self, widget=None):
        if not SCIPY_OK: return
        dlg = Gtk.FileChooserDialog(title="Zapisz WAV", parent=self,
                                    action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_SAVE,   Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK:
            wav.write(dlg.get_filename(), SR,
                      (self.current_wave * 32767).astype(np.int16))
            self._st(f"Zapisano: {dlg.get_filename()}")
        dlg.destroy()

    def _tmp(self, wave):
        p = os.path.join(tempfile.gettempdir(), "chaos_pad.wav")
        wav.write(p, SR, (wave * 32767).astype(np.int16)); return p

    def _on_destroy(self, widget):
        if self._timer_id:
            GLib.source_remove(self._timer_id)

    # ══════════════════════════════════════════════════════════════
    #  Podgląd i info
    # ══════════════════════════════════════════════════════════════
    def _update_info(self):
        p = self.current_params
        h  = p.get('h', [])
        a, d, s, r = p.get('adsr', [0,0,0,0])
        lines = [
            f"f0:  {p.get('f0', 0):.1f} Hz  (MIDI {p.get('midi_note', 69)})",
            f"poly: {p.get('poly', 1):.1f}  amb: {p.get('amb', 0):.2f}",
            f"H:  {' '.join(f'{v:.2f}' for v in h[:5])}",
            f"A={a:.2f} D={d:.2f} S={s:.2f} R={r:.2f}",
            f"noise: {p.get('noise', 0):.3f}",
        ]
        # strefa
        nx, ny = self.pad.nx, self.pad.ny
        ang = (math.degrees(math.atan2(nx, -ny)) + 360) % 360
        zi  = int(ang / 45) % 8
        zone_name = ZONES[zi + 1][1]
        r_val = math.sqrt(nx*nx + ny*ny)
        lines.insert(0, f"Strefa: {zone_name}  r={r_val:.2f}")
        self.info_label.set_text("\n".join(lines))

    def _draw_wave_preview(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        cr.set_source_rgb(0.08, 0.08, 0.10); cr.paint()
        wave = self.current_wave
        if len(wave) < 2: return
        step = len(wave) / w
        # strefa → kolor
        nx, ny = self.pad.nx, self.pad.ny
        ang = (math.degrees(math.atan2(nx, -ny)) + 360) % 360
        zi  = int(ang / 45) % 8
        rc, gc, bc = ZONES[zi + 1][2]
        cr.set_source_rgba(rc, gc, bc, 0.9)
        cr.set_line_width(1.2)
        first = True
        for i in range(w):
            idx = min(int(i * step), len(wave) - 1)
            y = h/2 - wave[idx] * (h/2 - 4)
            if first: cr.move_to(i, y); first = False
            else: cr.line_to(i, y)
        cr.stroke()
        # oś środkowa
        cr.set_source_rgba(0.5, 0.5, 0.55, 0.4)
        cr.set_line_width(0.5)
        cr.move_to(0, h/2); cr.line_to(w, h/2); cr.stroke()

    def _st(self, msg):
        self.status.set_text(msg)

    # ══════════════════════════════════════════════════════════════
    #  Publiczne API dla ED_Waves
    # ══════════════════════════════════════════════════════════════
    def get_current_wave(self):
        return self.current_wave.copy()

    def get_current_params(self):
        return self.current_params.copy()


# ═══════════════════════════════════════════════════════════════════
#  Standalone
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    def _demo_cb(wave, params):
        pass  # w standalone nic nie robimy — pad sam obsługuje play

    app = ChaosPadWindow(callback=_demo_cb)
    app.show_all()
    Gtk.main()
