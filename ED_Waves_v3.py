"""
ED_Waves v3
===========
• Każda fala (generowana / wczytana) dostaje automatyczne punkty kontrolne
  na zero-crossingach i ekstremach lokalnych.
• Modyfikacje (warstwy korekcji MUL/ADD) działają NA punktach kontrolnych
  → potem fala jest rekonstruowana ze zmodyfikowanych CP → opcjonalny DSP post.
• Profile dźwięczności: synth (lead/pad/bass/pluck/arp), perkusja (kick/snare/
  hat/tom/clap), efekty (distort/bitcrush/warm/bright).
• Panel prędkości (velocity) pod edytorem — naciągana linia.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
import random, tempfile, os, math

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ═══════════════════════════════════════════════════════════════════
#  Stałe UI
# ═══════════════════════════════════════════════════════════════════
RULER_H    = 28
FREQ_W     = 60
CP_R       = 5
SEL_R      = 9
FADE_W     = 48

LINE_BEZIER = "bezier"
LINE_LINEAR = "linear"
LINE_STEP   = "step"

CORR_MUL = "mul"
CORR_ADD = "add"

TOOL_DRAW   = "draw"
TOOL_EDIT   = "edit"
TOOL_SELECT = "select"

MAX_CP = 512


# ═══════════════════════════════════════════════════════════════════
#  Punkty kontrolne fali
# ═══════════════════════════════════════════════════════════════════
def extract_control_points(wave, max_pts=MAX_CP):
    n = len(wave)
    if n < 4:
        return [[0, float(wave[0])], [n-1, float(wave[-1])]]
    pts = {0, n-1}
    signs = np.sign(wave); signs[signs == 0] = 1
    for c in np.where(np.diff(signs))[0]: pts.add(int(c))
    for p in find_peaks( wave, distance=4)[0]: pts.add(int(p))
    for p in find_peaks(-wave, distance=4)[0]: pts.add(int(p))
    pts = sorted(pts)
    if len(pts) > max_pts:
        step = len(pts) / max_pts
        pts  = [pts[int(i*step)] for i in range(max_pts)]
        pts[0] = 0; pts[-1] = n-1
    return [[idx, float(wave[idx])] for idx in pts]


def reconstruct_wave(cps, n):
    if not cps: return np.zeros(n)
    idxs = np.array([c[0] for c in cps], dtype=float)
    amps = np.array([c[1] for c in cps], dtype=float)
    return np.interp(np.arange(n, dtype=float), idxs, amps)


# ═══════════════════════════════════════════════════════════════════
#  Warstwa korekcji
# ═══════════════════════════════════════════════════════════════════
class CorrectionLayer:
    MUL_MAX = 3.0

    def __init__(self, p0, p1, line_type=LINE_BEZIER, mode=CORR_MUL):
        self.p0 = list(p0); self.p1 = list(p1)
        self.line_type = line_type; self.mode = mode
        dx = (p1[0]-p0[0])/3.0
        self.cp0 = [p0[0]+dx, p0[1]]; self.cp1 = [p1[0]-dx, p1[1]]
        self.selected = False; self.enabled = True

    def _yn(self, py, oy, ch):
        return (oy + ch/2 - py) / max(ch/2, 1)

    def _curve_at(self, t_arr, oy, ch):
        y0  = self._yn(self.p0[1],  oy, ch)
        y1  = self._yn(self.p1[1],  oy, ch)
        yc0 = self._yn(self.cp0[1], oy, ch)
        yc1 = self._yn(self.cp1[1], oy, ch)
        ts  = t_arr
        if self.line_type == LINE_BEZIER:
            norm = ((1-ts)**3*y0 + 3*(1-ts)**2*ts*yc0
                    + 3*(1-ts)*ts**2*yc1 + ts**3*y1)
        elif self.line_type == LINE_LINEAR:
            norm = y0 + ts*(y1-y0)
        else:
            norm = np.where(ts < 0.5, y0, y1)
        if self.mode == CORR_MUL:
            return np.where(norm >= 0,
                            1.0 + norm*(self.MUL_MAX-1.0),
                            1.0 + norm)
        return norm

    def apply_to_cps(self, cps, ox, oy, cw, ch, n_wave):
        if not self.enabled or cw <= 0: return
        x0px = min(self.p0[0], self.p1[0])
        x1px = max(self.p0[0], self.p1[0])
        t0 = max(0.0, (x0px-ox)/cw); t1 = min(1.0, (x1px-ox)/cw)
        if t1 <= t0: return
        span = t1 - t0
        fade_frac = FADE_W / max(n_wave*span, 1)
        for cp in cps:
            t_g = cp[0] / max(n_wave-1, 1)
            if t_g < t0 or t_g > t1: continue
            t_loc = np.array([(t_g-t0)/max(span, 1e-9)])
            val   = self._curve_at(t_loc, oy, ch)[0]
            edge  = min((t_g-t0), (t1-t_g)) / max(span, 1e-9)
            w     = (0.5 - 0.5*math.cos(math.pi*edge/max(fade_frac,1e-9))
                     if edge < fade_frac else 1.0)
            if self.mode == CORR_MUL:
                cp[1] *= (1.0 + w*(val-1.0))
            else:
                cp[1] += w*val

    def draw(self, cr):
        sel = self.selected
        if self.mode == CORR_MUL:
            col = (0.25,0.85,1.00) if sel else (1.00,0.65,0.10)
        else:
            col = (0.85,0.35,1.00) if sel else (1.00,0.35,0.65)
        if not self.enabled: col = tuple(c*0.4 for c in col)
        cr.set_source_rgb(*col); cr.set_line_width(2.0 if sel else 1.6)
        x0,y0 = self.p0; x1,y1 = self.p1
        if self.line_type == LINE_BEZIER:
            cr.move_to(x0,y0)
            cr.curve_to(self.cp0[0],self.cp0[1],self.cp1[0],self.cp1[1],x1,y1)
            cr.stroke()
            if sel:
                cr.set_source_rgba(0.9,0.9,0.2,0.6); cr.set_line_width(1)
                cr.set_dash([4,4])
                cr.move_to(x0,y0); cr.line_to(self.cp0[0],self.cp0[1]); cr.stroke()
                cr.move_to(x1,y1); cr.line_to(self.cp1[0],self.cp1[1]); cr.stroke()
                cr.set_dash([])
                for px,py in [self.cp0,self.cp1]:
                    cr.set_source_rgba(0.95,0.90,0.20,0.90)
                    cr.arc(px,py,4,0,6.28); cr.fill()
        elif self.line_type == LINE_LINEAR:
            cr.move_to(x0,y0); cr.line_to(x1,y1); cr.stroke()
        else:
            mx=(x0+x1)/2
            cr.move_to(x0,y0); cr.line_to(mx,y0)
            cr.line_to(mx,y1); cr.line_to(x1,y1); cr.stroke()
        for px,py in [(x0,y0),(x1,y1)]:
            cr.set_source_rgb(*col); cr.arc(px,py,CP_R,0,6.28); cr.fill()
            cr.set_source_rgb(0,0,0); cr.arc(px,py,CP_R,0,6.28)
            cr.set_line_width(1); cr.stroke()
        cr.set_source_rgba(*col,0.85); cr.set_font_size(9)
        cr.move_to((x0+x1)/2-8, min(y0,y1)-8)
        cr.show_text(self.mode.upper())


# ═══════════════════════════════════════════════════════════════════
#  DSP helpers
# ═══════════════════════════════════════════════════════════════════
def _normalize(w):
    mx = np.max(np.abs(w)); return w/mx if mx > 1e-9 else w

def _adsr(n, a, d, s_lv, r, sr):
    env=np.ones(n)
    ia=min(int(a*sr),n); id_=min(int(d*sr),n-ia); ir=min(int(r*sr),n)
    if ia: env[:ia]=np.linspace(0,1,ia)
    if id_: env[ia:ia+id_]=np.linspace(1,s_lv,id_)
    env[ia+id_:n-ir]=s_lv
    if ir and n-ir>=0: env[n-ir:]=np.linspace(s_lv,0,ir)
    return env

def _lp(wave, cut, sr, poles=1):
    from scipy.signal import butter, sosfilt
    nyq=sr/2
    if cut>=nyq: return wave
    sos=butter(poles,cut/nyq,btype='low',output='sos')
    return sosfilt(sos,wave)

def _hp(wave, cut, sr, poles=1):
    from scipy.signal import butter, sosfilt
    nyq=sr/2
    if cut<=0: return wave
    sos=butter(poles,max(cut/nyq,1e-4),btype='high',output='sos')
    return sosfilt(sos,wave)

def _soft(wave, drive=1.0):
    return np.tanh(wave*drive)/math.tanh(drive) if drive>0 else wave

def _bitcrush(wave, bits):
    s=2**max(1,int(bits)); return np.round(wave*s)/s


# ═══════════════════════════════════════════════════════════════════
#  Profile dźwięczności
# ═══════════════════════════════════════════════════════════════════
class SoundProfile:
    PROFILES = {}

    @staticmethod
    def apply_cps(name, cps, n):
        fn = SoundProfile.PROFILES.get(name,{}).get('cps')
        if fn: fn(cps, n)

    @staticmethod
    def apply_dsp(name, wave, sr):
        fn = SoundProfile.PROFILES.get(name,{}).get('dsp')
        return fn(wave, sr) if fn else wave

    @staticmethod
    def names(): return list(SoundProfile.PROFILES.keys())

def _reg(name, cps_fn=None, dsp_fn=None):
    SoundProfile.PROFILES[name] = {'cps': cps_fn, 'dsp': dsp_fn}

SR = 44100

# lead
def _lead_c(cps,n):
    env=_adsr(n,0.005,0.05,0.8,0.1,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("lead", _lead_c, lambda w,sr: _normalize(_soft(w,1.4)))

# pad
def _pad_c(cps,n):
    env=_adsr(n,0.3,0.1,0.9,0.5,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("pad", _pad_c, lambda w,sr: _normalize(_lp(w,4000,sr,2)))

# bass
def _bass_c(cps,n):
    env=_adsr(n,0.01,0.08,0.85,0.15,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("bass", _bass_c, lambda w,sr: _normalize(_soft(_lp(w,2500,sr,2),1.8)))

# pluck
def _pluck_c(cps,n):
    env=_adsr(n,0.002,0.25,0.0,0.05,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("pluck", _pluck_c, lambda w,sr: _normalize(_hp(w,80,sr)))

# arp
def _arp_c(cps,n):
    gate=np.ones(n); gate[n//2:]=0.0
    for cp in cps: cp[1]*=gate[min(cp[0],n-1)]
_reg("arp", _arp_c, None)

# kick
def _kick_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*18)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("kick", _kick_c, lambda w,sr: _normalize(_soft(_lp(w,120,sr,2)*1.5,2.0)))

# snare
def _snare_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*30)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
def _snare_d(w,sr):
    noise=np.random.normal(0,0.3,len(w))*np.exp(-np.arange(len(w))/sr*25)
    return _normalize(_hp(w+noise,200,sr))
_reg("snare", _snare_c, _snare_d)

# hat
def _hat_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*80)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("hat", _hat_c, lambda w,sr: _normalize(_hp(w,6000,sr,2)))

# tom
def _tom_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*12)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("tom", _tom_c, lambda w,sr: _normalize(_lp(_soft(w,1.2),800,sr,2)))

# clap
def _clap_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*40)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
def _clap_d(w,sr):
    noise=np.random.normal(0,1.0,len(w))*np.exp(-np.arange(len(w))/sr*40)
    return _normalize(_hp(w*0.3+noise*0.7,500,sr))
_reg("clap", _clap_c, _clap_d)

# efekty — tylko DSP
_reg("distort",  None, lambda w,sr: _normalize(_soft(w,4.0)))
_reg("bitcrush", None, lambda w,sr: _normalize(_bitcrush(w,5)))
_reg("warm",     None, lambda w,sr: _normalize(_lp(w,8000,sr,2)*1.1))
def _bright_d(w,sr):
    from scipy.signal import butter,sosfilt
    nyq=sr/2; sos=butter(1,min(5000/nyq,0.99),btype='high',output='sos')
    return _normalize(w+sosfilt(sos,w)*0.6)
_reg("bright", None, _bright_d)


# ═══════════════════════════════════════════════════════════════════
#  Generator bazowych fal
# ═══════════════════════════════════════════════════════════════════
def generate_base_wave(wtype, duration, sr=44100):
    n = int(duration*sr)
    t = np.linspace(0, duration, n, endpoint=False)
    if   wtype=="sine":        w=np.sin(2*np.pi*440*t)
    elif wtype=="saw":         f=220; w=2*(t*f-np.floor(t*f+0.5))
    elif wtype=="square":      w=np.sign(np.sin(2*np.pi*330*t))
    elif wtype=="triangle":    f=330; w=2*np.abs(2*(t*f-np.floor(t*f+0.5)))-1
    elif wtype=="noise_white": w=np.random.normal(0,1,n)
    elif wtype=="noise_pink":  w=_lp(np.random.normal(0,1,n),2000,sr,2)
    elif wtype=="fm":
        fc=random.uniform(200,600); fm=random.uniform(2,8); idx=random.uniform(1,5)
        w=np.sin(2*np.pi*fc*t+idx*np.sin(2*np.pi*fc*fm*t))
    elif wtype=="additive":
        f0=random.uniform(110,440); h=random.randint(3,8)
        w=sum(np.sin(2*np.pi*f0*(i+1)*t)/(i+1)**random.uniform(0.8,1.5)
              for i in range(h))
    else: w=np.zeros(n)
    return _normalize(w)


# ═══════════════════════════════════════════════════════════════════
#  Główna klasa
# ═══════════════════════════════════════════════════════════════════
class WaveformEditor(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="ED_Waves v3")
        self.set_default_size(1400, 820)
        self.connect("destroy", Gtk.main_quit)

        self.sample_rate  = 44100
        self.zoom_level   = 1.0
        self.view_start   = 0
        self.view_end     = 0

        self.original_wave = np.array([])
        self.wave_cps      = []
        self.waveform      = np.array([])

        self.layers        = []
        self.draw_start    = None
        self.mouse_pos     = (0,0)
        self.drag_target   = None
        self.drag_offset   = (0,0)
        self.sel_rect_s    = None
        self.sel_rect_e    = None

        self.current_tool  = TOOL_DRAW
        self.current_line  = LINE_BEZIER
        self.current_corr  = CORR_MUL
        self.show_cps      = True
        self.sample_w_px   = 2

        self.pre_profile   = None
        self.post_profile  = None

        self.vel_nodes     = [[0.0,1.0],[1.0,1.0]]
        self.vel_drag_idx  = None
        self.vel_apply     = True
        self.live_play     = False

        self._build_ui()

    # ════════════════════════ UI ══════════════════════════════════
    def _build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add(root)

        # ── pasek 1: narzędzia ────────────────────────────────────
        tb1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        tb1.set_border_width(3); root.pack_start(tb1,False,False,0)

        tg=None
        for lbl,t in [("✏ Draw",TOOL_DRAW),("✋ Edit",TOOL_EDIT),("⬚ Select",TOOL_SELECT)]:
            b=Gtk.RadioButton.new_with_label_from_widget(tg,lbl)
            if tg is None: tg=b
            b.connect("toggled",self._on_tool,t); tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        lg=None
        for lbl,lt in [("~ Bézier",LINE_BEZIER),("/ Linia",LINE_LINEAR),("⌐ Schodek",LINE_STEP)]:
            b=Gtk.RadioButton.new_with_label_from_widget(lg,lbl)
            if lg is None: lg=b
            b.connect("toggled",self._on_linetype,lt); tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        cg=None
        for lbl,cm in [("MUL ×",CORR_MUL),("ADD ±",CORR_ADD)]:
            b=Gtk.RadioButton.new_with_label_from_widget(cg,lbl)
            if cg is None: cg=b
            b.connect("toggled",self._on_corrmode,cm); tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        for lbl,cb in [("▶",self.on_play),("⏹",self.on_stop),("⟳ Live",self._toggle_live),
                       ("💾 WAV",self.on_save_wav),("📂 Import",self.on_import_wav),
                       ("🗑 Warstwy",self._clear_layers)]:
            b=Gtk.Button(label=lbl); b.connect("clicked",cb); tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        tb1.pack_start(Gtk.Label(label="px/próbka:"),False,False,0)
        self.sw_spin=Gtk.SpinButton(); self.sw_spin.set_range(1,20); self.sw_spin.set_value(2)
        self.sw_spin.set_increments(1,2)
        self.sw_spin.connect("value-changed",lambda s:setattr(self,'sample_w_px',int(s.get_value())))
        tb1.pack_start(self.sw_spin,False,False,0)

        chk=Gtk.CheckButton(label="Pokaż CP"); chk.set_active(True)
        chk.connect("toggled",lambda w:[setattr(self,'show_cps',w.get_active()),
                                        self.drawing_area.queue_draw()])
        tb1.pack_start(chk,False,False,0)

        # ── pasek 2: generator + profile ─────────────────────────
        tb2=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=4)
        tb2.set_border_width(2); root.pack_start(tb2,False,False,0)

        tb2.pack_start(Gtk.Label(label="Fala:"),False,False,0)
        self.gen_combo=Gtk.ComboBoxText()
        for wt in ["sine","saw","square","triangle","noise_white","noise_pink","fm","additive"]:
            self.gen_combo.append_text(wt)
        self.gen_combo.set_active(0); tb2.pack_start(self.gen_combo,False,False,0)

        tb2.pack_start(Gtk.Label(label="s:"),False,False,0)
        self.gen_dur=Gtk.SpinButton(); self.gen_dur.set_range(0.05,5.0)
        self.gen_dur.set_value(1.0); self.gen_dur.set_increments(0.05,0.5)
        self.gen_dur.set_digits(2); tb2.pack_start(self.gen_dur,False,False,0)

        bg=Gtk.Button(label="⚡ Generuj"); bg.connect("clicked",self._on_generate)
        tb2.pack_start(bg,False,False,0)

        tb2.pack_start(_sep(),False,False,6)
        tb2.pack_start(Gtk.Label(label="Pre-profil:"),False,False,0)
        self.pre_combo=Gtk.ComboBoxText(); self.pre_combo.append_text("(brak)")
        for p in SoundProfile.names(): self.pre_combo.append_text(p)
        self.pre_combo.set_active(0); tb2.pack_start(self.pre_combo,False,False,0)
        bp=Gtk.Button(label="▶ Pre"); bp.connect("clicked",self._apply_pre); tb2.pack_start(bp,False,False,0)

        tb2.pack_start(_sep(),False,False,4)
        tb2.pack_start(Gtk.Label(label="Post-profil:"),False,False,0)
        self.post_combo=Gtk.ComboBoxText(); self.post_combo.append_text("(brak)")
        for p in SoundProfile.names(): self.post_combo.append_text(p)
        self.post_combo.set_active(0); tb2.pack_start(self.post_combo,False,False,0)
        bp2=Gtk.Button(label="▶ Post"); bp2.connect("clicked",self._apply_post); tb2.pack_start(bp2,False,False,0)

        br=Gtk.Button(label="↩ Reset"); br.connect("clicked",self._reset_all); tb2.pack_start(br,False,False,0)

        # ── pasek 3: transformacje ────────────────────────────────
        tb3=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=4)
        tb3.set_border_width(2); root.pack_start(tb3,False,False,0)
        tb3.pack_start(Gtk.Label(label="Zaznaczenie:"),False,False,0)
        for lbl,act in [("Ścisn.Y",'compress_y'),("Rozciąg.Y",'stretch_y'),
                        ("Ścisn.X",'compress_x'),("Rozciąg.X",'stretch_x'),
                        ("Odwróć Y",'flip_y'),("Odwróć X",'flip_x'),
                        ("Wł/Wył",'toggle'),("Usuń",'delete')]:
            b=Gtk.Button(label=lbl)
            b.connect("clicked",lambda w,a=act:self._transform_sel(a))
            tb3.pack_start(b,False,False,0)
        tb3.pack_start(Gtk.Label(label=" sY:"),False,False,0)
        self.sel_sy=Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_sy.set_range(0.05,4.0); self.sel_sy.set_value(1.0)
        self.sel_sy.set_digits(2); self.sel_sy.set_size_request(100,-1)
        tb3.pack_start(self.sel_sy,False,False,0)
        tb3.pack_start(Gtk.Label(label=" sX:"),False,False,0)
        self.sel_sx=Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_sx.set_range(0.05,4.0); self.sel_sx.set_value(1.0)
        self.sel_sx.set_digits(2); self.sel_sx.set_size_request(100,-1)
        tb3.pack_start(self.sel_sx,False,False,0)

        # ── drawing area ──────────────────────────────────────────
        hb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        root.pack_start(hb,True,True,0)
        self.drawing_area=Gtk.DrawingArea(); self.drawing_area.set_size_request(1280,420)
        self.drawing_area.connect("draw",self.on_draw)
        self.drawing_area.connect("button-press-event",  self.on_press)
        self.drawing_area.connect("button-release-event",self.on_release)
        self.drawing_area.connect("motion-notify-event", self.on_motion)
        mask=(Gdk.EventMask.BUTTON_PRESS_MASK|Gdk.EventMask.BUTTON_RELEASE_MASK|
              Gdk.EventMask.POINTER_MOTION_MASK)
        self.drawing_area.set_events(mask); hb.pack_start(self.drawing_area,True,True,0)
        self.vscroll=Gtk.VScrollbar(); self.vscroll.connect("value-changed",self._on_scroll)
        hb.pack_start(self.vscroll,False,False,0)

        # ── velocity ──────────────────────────────────────────────
        vf=Gtk.Frame(label=" Velocity (LPM=węzeł, PPM=usuń, naciągnij ↑↓) ")
        vf.set_border_width(2); root.pack_start(vf,False,False,0)
        vvb=Gtk.Box(orientation=Gtk.Orientation.VERTICAL); vf.add(vvb)
        vtb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=5)
        vtb.set_border_width(2); vvb.pack_start(vtb,False,False,0)
        vchk=Gtk.CheckButton(label="Stosuj velocity"); vchk.set_active(True)
        vchk.connect("toggled",lambda w:[setattr(self,'vel_apply',w.get_active()),self._rerender()])
        vtb.pack_start(vchk,False,False,0)
        vrst=Gtk.Button(label="Reset 100%"); vrst.connect("clicked",self._vel_reset)
        vtb.pack_start(vrst,False,False,0)
        self.vel_area=Gtk.DrawingArea(); self.vel_area.set_size_request(-1,70)
        self.vel_area.connect("draw",self._vel_draw)
        self.vel_area.connect("button-press-event",  self._vel_press)
        self.vel_area.connect("button-release-event",self._vel_release)
        self.vel_area.connect("motion-notify-event", self._vel_motion)
        self.vel_area.set_events(mask); vvb.pack_start(self.vel_area,True,True,0)

        # ── zoom ──────────────────────────────────────────────────
        zb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=4)
        zb.set_border_width(2); root.pack_start(zb,False,False,0)
        zb.pack_start(Gtk.Label(label="Zoom:"),False,False,0)
        self.zoom_sc=Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.zoom_sc.set_range(1,20); self.zoom_sc.set_value(1)
        self.zoom_sc.set_digits(1); self.zoom_sc.set_increments(0.1,1)
        self.zoom_sc.set_hexpand(True)
        self.zoom_sc.connect("value-changed",lambda w:[setattr(self,'zoom_level',w.get_value()),
                             self._update_view(), self.drawing_area.queue_draw()])
        zb.pack_start(self.zoom_sc,True,True,0)

        # ── status ────────────────────────────────────────────────
        self.status=Gtk.Label(label="Gotowy — wygeneruj falę lub wczytaj WAV")
        self.status.set_xalign(0); self.status.set_margin_start(4)
        self.status.set_margin_top(2); self.status.set_margin_bottom(2)
        root.pack_start(self.status,False,False,0)

    # ══════════════════════════════════════════════════════════════
    #  Callbacki narzędzi
    # ══════════════════════════════════════════════════════════════
    def _on_tool(self,btn,t):
        if btn.get_active(): self.current_tool=t
    def _on_linetype(self,btn,lt):
        if btn.get_active():
            self.current_line=lt
            for l in self.layers:
                if l.selected: l.line_type=lt
            self.drawing_area.queue_draw()
    def _on_corrmode(self,btn,cm):
        if btn.get_active(): self.current_corr=cm

    # ══════════════════════════════════════════════════════════════
    #  Generator
    # ══════════════════════════════════════════════════════════════
    def _on_generate(self, widget):
        wtype = self.gen_combo.get_active_text() or "sine"
        dur   = self.gen_dur.get_value()
        wave  = generate_base_wave(wtype, dur, self.sample_rate)
        self._set_wave(wave)
        self._st(f"Wygenerowano: {wtype}  {dur:.2f}s  {len(wave)} próbek  CP={len(self.wave_cps)}")

    def _set_wave(self, wave):
        self.original_wave = wave.copy()
        self.wave_cps      = extract_control_points(wave)
        self.layers        = []
        self.view_start    = 0; self.view_end = len(wave)
        self._rerender(); self._update_view()
        self.drawing_area.queue_draw(); self.vel_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Profile
    # ══════════════════════════════════════════════════════════════
    def _pname(self, combo):
        t=combo.get_active_text(); return None if t=="(brak)" else t

    def _apply_pre(self, widget):
        name=self._pname(self.pre_combo)
        if not name or len(self.original_wave)<2: return
        cps=[list(c) for c in self.wave_cps]
        SoundProfile.apply_cps(name, cps, len(self.original_wave))
        mx=max((abs(c[1]) for c in cps), default=1.0)
        if mx>1e-9:
            for c in cps: c[1]/=mx
        new_wave=reconstruct_wave(cps, len(self.original_wave))
        new_wave=SoundProfile.apply_dsp(name, new_wave, self.sample_rate)
        new_wave=_normalize(new_wave)
        # pre modyfikuje oryginał → nowe CP
        self.original_wave=new_wave
        self.wave_cps=extract_control_points(new_wave)
        self._rerender(); self.drawing_area.queue_draw()
        self._st(f"Pre-profil: {name}  CP={len(self.wave_cps)}")

    def _apply_post(self, widget):
        name=self._pname(self.post_combo)
        if not name or len(self.waveform)<2: return
        w=SoundProfile.apply_dsp(name, self.waveform.copy(), self.sample_rate)
        self.waveform=_normalize(w)
        self.drawing_area.queue_draw()
        if self.live_play: self._play_now()
        self._st(f"Post DSP: {name}")

    def _reset_all(self, widget):
        self.wave_cps=extract_control_points(self.original_wave)
        self.layers=[]; self._rerender()
        self.drawing_area.queue_draw()
        self._st("Reset — oryginalna fala + CP odtworzone")

    # ══════════════════════════════════════════════════════════════
    #  Renderowanie: CP + warstwy + velocity → waveform
    # ══════════════════════════════════════════════════════════════
    def _rerender(self):
        if len(self.original_wave)<2: return
        ox,oy,cw,ch=self._crect()
        n=len(self.original_wave)
        cps=[list(c) for c in self.wave_cps]
        for layer in self.layers:
            if layer.enabled:
                layer.apply_to_cps(cps, ox, oy, cw, ch, n)
        wave=reconstruct_wave(cps, n)
        wave=_normalize(wave) if np.max(np.abs(wave))>1e-9 else wave
        if self.vel_apply:
            wave=wave*self._vel_curve(n)
        self.waveform=wave
        if self.live_play: self._play_now()

    # ══════════════════════════════════════════════════════════════
    #  on_draw
    # ══════════════════════════════════════════════════════════════
    def on_draw(self, widget, cr):
        ox,oy,cw,ch=self._crect()
        cr.set_source_rgb(0.10,0.10,0.12); cr.paint()
        cr.set_source_rgb(0.16,0.16,0.19); cr.rectangle(0,oy,FREQ_W,ch); cr.fill()
        self._draw_freq_axis(cr,oy,ch)
        self._draw_ruler(cr,ox,oy,cw)
        self._draw_grid(cr,ox,oy,cw,ch)
        cr.set_source_rgba(0.5,0.5,0.5,0.6); cr.set_line_width(1)
        cr.move_to(ox,oy+ch/2); cr.line_to(ox+cw,oy+ch/2); cr.stroke()
        if len(self.waveform)>1: self._draw_wave(cr,ox,oy,cw,ch)
        if self.show_cps and self.wave_cps: self._draw_wave_cps(cr,ox,oy,cw,ch)
        for layer in self.layers: layer.draw(cr)
        if self.draw_start and self.current_tool==TOOL_DRAW:
            cr.set_source_rgba(0.3,0.9,1.0,0.45); cr.set_line_width(1.5); cr.set_dash([6,4])
            cr.move_to(*self.draw_start); cr.line_to(*self.mouse_pos); cr.stroke()
            cr.set_dash([]); cr.set_source_rgba(0.3,0.9,1.0,0.85)
            cr.arc(self.draw_start[0],self.draw_start[1],CP_R+1,0,6.28); cr.fill()
        if self.sel_rect_s and self.sel_rect_e:
            x1,y1=self.sel_rect_s; x2,y2=self.sel_rect_e
            rx,ry=min(x1,x2),min(y1,y2); rw,rh=abs(x2-x1),abs(y2-y1)
            cr.set_source_rgba(0.3,0.75,1.0,0.12); cr.rectangle(rx,ry,rw,rh); cr.fill()
            cr.set_source_rgba(0.3,0.75,1.0,0.75); cr.set_line_width(1)
            cr.rectangle(rx,ry,rw,rh); cr.stroke()

    def _draw_wave(self, cr, ox, oy, cw, ch):
        cr.set_source_rgba(0.20,0.72,0.32,0.85); cr.set_line_width(1.2)
        spx=max(1,self.sample_w_px); pts=int(cw//spx)
        if pts<2: return
        vlen=max(1,self.view_end-self.view_start); first=True
        for i in range(pts):
            idx=int(self.view_start+(i/pts)*vlen)
            if idx>=len(self.waveform): break
            x=ox+i*spx; y=oy+ch/2-self.waveform[idx]*ch/2
            if first: cr.move_to(x,y); first=False
            else: cr.line_to(x,y)
        cr.stroke()

    def _draw_wave_cps(self, cr, ox, oy, cw, ch):
        n=len(self.original_wave); vlen=max(1,self.view_end-self.view_start)
        for cp in self.wave_cps:
            idx,amp=cp[0],cp[1]
            t_view=(idx-self.view_start)/vlen
            if t_view<-0.01 or t_view>1.01: continue
            x=ox+t_view*cw; y=oy+ch/2-amp*ch/2
            cr.set_source_rgba(0.9,0.5,0.1,0.75)
            cr.arc(x,y,3,0,6.28); cr.fill()

    def _draw_freq_axis(self, cr, oy, ch):
        cr.set_font_size(9)
        for f in [20,50,100,200,500,1000,2000,5000,10000,20000]:
            pos=(math.log10(f)-math.log10(5))/(math.log10(25000)-math.log10(5))
            y=oy+ch-pos*ch
            cr.set_source_rgba(0.5,0.5,0.55,0.65)
            cr.move_to(0,y); cr.line_to(FREQ_W,y); cr.set_line_width(0.5); cr.stroke()
            cr.set_source_rgb(0.78,0.78,0.78); cr.move_to(2,y-2)
            cr.show_text(f"{f}Hz" if f<1000 else f"{f//1000}k")

    def _draw_ruler(self, cr, ox, oy, cw):
        cr.set_source_rgb(0.15,0.15,0.18); cr.rectangle(ox,0,cw,RULER_H); cr.fill()
        if len(self.waveform)<2: return
        vlen=max(1,self.view_end-self.view_start); nticks=max(1,cw//90)
        cr.set_font_size(9); cr.set_source_rgb(0.70,0.70,0.70)
        for i in range(nticks+1):
            x=ox+i*cw/nticks; samp=self.view_start+i*vlen/nticks
            ms=samp/self.sample_rate*1000
            cr.move_to(x,RULER_H-7); cr.line_to(x,RULER_H); cr.set_line_width(1); cr.stroke()
            cr.move_to(x+2,RULER_H-3); cr.show_text(f"{ms:.0f}ms")

    def _draw_grid(self, cr, ox, oy, cw, ch):
        cr.set_source_rgba(0.22,0.22,0.25,0.45); cr.set_line_width(0.5)
        for i in range(1,4):
            y=oy+i*ch/4; cr.move_to(ox,y); cr.line_to(ox+cw,y); cr.stroke()
        for i in range(1,8):
            x=ox+i*cw/8; cr.move_to(x,oy); cr.line_to(x,oy+ch); cr.stroke()

    # ══════════════════════════════════════════════════════════════
    #  Mysz
    # ══════════════════════════════════════════════════════════════
    def on_press(self, widget, event):
        x,y=event.x,event.y
        if event.button==1:
            if   self.current_tool==TOOL_DRAW:   self.draw_start=(x,y)
            elif self.current_tool==TOOL_EDIT:   self._edit_press(x,y)
            elif self.current_tool==TOOL_SELECT: self._sel_press(x,y,event)
        elif event.button==3: self._ctx_menu(x,y,event)

    def on_release(self, widget, event):
        x,y=event.x,event.y
        if event.button==1:
            if   self.current_tool==TOOL_DRAW:   self._draw_release(x,y)
            elif self.current_tool==TOOL_EDIT:   self.drag_target=None
            elif self.current_tool==TOOL_SELECT: self._sel_release(x,y)
        self.drawing_area.queue_draw(); self._rerender()

    def on_motion(self, widget, event):
        x,y=event.x,event.y; self.mouse_pos=(x,y)
        if self.current_tool==TOOL_EDIT and self.drag_target: self._edit_move(x,y)
        elif self.current_tool==TOOL_SELECT and self.sel_rect_s: self.sel_rect_e=(x,y)
        self.drawing_area.queue_draw()
        ox,oy,cw,ch=self._crect()
        t=(x-ox)/max(cw,1); a=(oy+ch/2-y)/max(ch/2,1)
        if len(self.waveform)>0:
            vlen=max(1,self.view_end-self.view_start)
            idx=int(self.view_start+t*vlen); ms=max(idx,0)/self.sample_rate*1000
            self._st(f"t={ms:.1f}ms  amp={a:.3f}  CP={len(self.wave_cps)}  "
                     f"warstwy={len(self.layers)}  narz={self.current_tool}  "
                     f"tryb={self.current_corr}")

    def _draw_release(self, x, y):
        if self.draw_start is None: return
        x0,y0=self.draw_start; self.draw_start=None
        if abs(x-x0)<5 and abs(y-y0)<5: return
        self.layers.append(CorrectionLayer([x0,y0],[x,y],self.current_line,self.current_corr))
        self.drawing_area.queue_draw()

    def _edit_press(self, x, y):
        best=None; best_d=SEL_R**2
        for i,l in enumerate(self.layers):
            for attr in ["p0","p1","cp0","cp1"]:
                if attr in ["cp0","cp1"] and not l.selected: continue
                pt=getattr(l,attr); d=(pt[0]-x)**2+(pt[1]-y)**2
                if d<best_d: best_d=d; best=(i,attr)
        if best:
            self.drag_target=best; pt=getattr(self.layers[best[0]],best[1])
            self.drag_offset=(x-pt[0],y-pt[1])

    def _edit_move(self, x, y):
        if not self.drag_target: return
        i,attr=self.drag_target; l=self.layers[i]
        nx=x-self.drag_offset[0]; ny=y-self.drag_offset[1]
        pt=getattr(l,attr); dx,dy=nx-pt[0],ny-pt[1]
        pt[0]=nx; pt[1]=ny
        if attr=="p0": l.cp0[0]+=dx; l.cp0[1]+=dy
        if attr=="p1": l.cp1[0]+=dx; l.cp1[1]+=dy

    def _sel_press(self, x, y, event):
        if event.state&Gdk.ModifierType.CONTROL_MASK:
            l=self._layer_at(x,y)
            if l: l.selected=not l.selected
        else:
            for l in self.layers: l.selected=False
            self.sel_rect_s=(x,y); self.sel_rect_e=(x,y)
        self.drawing_area.queue_draw()

    def _sel_release(self, x, y):
        if self.sel_rect_s:
            x1,y1=self.sel_rect_s
            rx0,rx1=min(x1,x),max(x1,x); ry0,ry1=min(y1,y),max(y1,y)
            for l in self.layers:
                px,py=l.p0
                if rx0<=px<=rx1 and ry0<=py<=ry1: l.selected=True
                px,py=l.p1
                if rx0<=px<=rx1 and ry0<=py<=ry1: l.selected=True
            self.sel_rect_s=None; self.sel_rect_e=None

    def _layer_at(self, x, y, thresh=14):
        for l in reversed(self.layers):
            mx=(l.p0[0]+l.p1[0])/2; my=(l.p0[1]+l.p1[1])/2
            if abs(mx-x)<thresh*3 and abs(my-y)<thresh*3: return l
        return None

    def _transform_sel(self, action):
        sel=[l for l in self.layers if l.selected]
        if not sel: self._st("Brak zaznaczonych!"); return
        sy=self.sel_sy.get_value(); sx=self.sel_sx.get_value()
        xs=[c for l in sel for c in [l.p0[0],l.p1[0]]]
        ys=[c for l in sel for c in [l.p0[1],l.p1[1]]]
        cx=sum(xs)/len(xs); cy=sum(ys)/len(ys)
        def sc(pt,scx,scy): pt[0]=cx+(pt[0]-cx)*scx; pt[1]=cy+(pt[1]-cy)*scy
        for l in sel:
            pts=[l.p0,l.p1,l.cp0,l.cp1]
            if   action=='compress_y': [sc(p,1,1/max(sy,0.01)) for p in pts]
            elif action=='stretch_y':  [sc(p,1,sy) for p in pts]
            elif action=='compress_x': [sc(p,1/max(sx,0.01),1) for p in pts]
            elif action=='stretch_x':  [sc(p,sx,1) for p in pts]
            elif action=='flip_y':     [p.__setitem__(1,2*cy-p[1]) for p in pts]
            elif action=='flip_x':     [p.__setitem__(0,2*cx-p[0]) for p in pts]
            elif action=='toggle':     l.enabled=not l.enabled
            elif action=='delete':
                if l in self.layers: self.layers.remove(l)
        self.drawing_area.queue_draw(); self._rerender()

    def _ctx_menu(self, x, y, event):
        l=self._layer_at(x,y); menu=Gtk.Menu()
        if l:
            for lbl,lt in [("Bézier",LINE_BEZIER),("Liniowy",LINE_LINEAR),("Schodek",LINE_STEP)]:
                item=Gtk.MenuItem(label=f"→ {lbl}")
                item.connect("activate",lambda w,s=l,t=lt:[setattr(s,'line_type',t),
                             self.drawing_area.queue_draw(),self._rerender()])
                menu.append(item)
            for lbl,cm in [("MUL ×",CORR_MUL),("ADD ±",CORR_ADD)]:
                item=Gtk.MenuItem(label=f"Tryb: {lbl}")
                item.connect("activate",lambda w,s=l,c=cm:[setattr(s,'mode',c),
                             self.drawing_area.queue_draw(),self._rerender()])
                menu.append(item)
            menu.append(Gtk.SeparatorMenuItem())
            item=Gtk.MenuItem(label="Usuń warstwę")
            item.connect("activate",lambda w,s=l:[self.layers.remove(s) if s in self.layers else None,
                         self.drawing_area.queue_draw(),self._rerender()])
            menu.append(item)
        else:
            item=Gtk.MenuItem(label="Wyczyść warstwy")
            item.connect("activate",self._clear_layers)
            menu.append(item)
        menu.show_all(); menu.popup_at_pointer(event)

    def _clear_layers(self, w=None):
        self.layers.clear(); self.drawing_area.queue_draw(); self._rerender()

    # ══════════════════════════════════════════════════════════════
    #  Velocity
    # ══════════════════════════════════════════════════════════════
    VEL_PL=FREQ_W; VEL_PR=14

    def _vrect(self):
        w=self.vel_area.get_allocated_width(); h=self.vel_area.get_allocated_height()
        ox=self.VEL_PL; return ox,4,w-ox-self.VEL_PR,h-8

    def _vt2x(self,t): ox,oy,cw,ch=self._vrect(); return ox+t*cw
    def _vv2y(self,v): ox,oy,cw,ch=self._vrect(); return oy+(1-v)*ch
    def _vx2t(self,x): ox,oy,cw,ch=self._vrect(); return max(0.,min(1.,(x-ox)/max(cw,1)))
    def _vy2v(self,y): ox,oy,cw,ch=self._vrect(); return max(0.,min(1.,1.-(y-oy)/max(ch,1)))

    def _vel_draw(self, widget, cr):
        ox,oy,cw,ch=self._vrect()
        cr.set_source_rgb(0.10,0.10,0.12); cr.paint()
        cr.set_source_rgb(0.14,0.14,0.17); cr.rectangle(ox,oy,cw,ch); cr.fill()
        cr.set_font_size(8)
        for pct in [0,25,50,75,100]:
            v=pct/100.; y=self._vv2y(v)
            cr.set_source_rgba(0.3,0.3,0.35,0.5); cr.set_line_width(0.5)
            cr.move_to(ox,y); cr.line_to(ox+cw,y); cr.stroke()
            cr.set_source_rgb(0.55,0.55,0.55); cr.move_to(2,y+3); cr.show_text(f"{pct}%")
        nodes=sorted(self.vel_nodes,key=lambda n:n[0])
        if nodes:
            cr.set_source_rgba(0.25,0.65,0.40,0.18)
            cr.move_to(self._vt2x(nodes[0][0]),oy+ch)
            cr.line_to(self._vt2x(nodes[0][0]),self._vv2y(nodes[0][1]))
            for nd in nodes[1:]: cr.line_to(self._vt2x(nd[0]),self._vv2y(nd[1]))
            cr.line_to(self._vt2x(nodes[-1][0]),oy+ch); cr.close_path(); cr.fill()
        cr.set_source_rgb(0.30,0.85,0.50); cr.set_line_width(1.8); first=True
        for nd in nodes:
            x=self._vt2x(nd[0]); y=self._vv2y(nd[1])
            if first: cr.move_to(x,y); first=False
            else: cr.line_to(x,y)
        cr.stroke()
        for i,nd in enumerate(self.vel_nodes):
            x=self._vt2x(nd[0]); y=self._vv2y(nd[1])
            cr.set_source_rgb(1.0,0.85,0.1) if i==self.vel_drag_idx else cr.set_source_rgb(0.3,0.9,0.5)
            cr.arc(x,y,5,0,6.28); cr.fill()
            cr.set_source_rgb(0,0,0); cr.arc(x,y,5,0,6.28); cr.set_line_width(1); cr.stroke()
        if not self.vel_apply:
            cr.set_source_rgba(0.8,0.2,0.2,0.55); cr.set_font_size(11)
            cr.move_to(ox+cw/2-40,oy+ch/2+5); cr.show_text("VELOCITY OFF")

    def _vel_nearest(self, x, y, thresh=12):
        best=None; best_d=thresh**2
        for i,nd in enumerate(self.vel_nodes):
            nx=self._vt2x(nd[0]); ny=self._vv2y(nd[1]); d=(nx-x)**2+(ny-y)**2
            if d<best_d: best_d=d; best=i
        return best

    def _vel_press(self, widget, event):
        x,y=event.x,event.y
        if event.button==3:
            idx=self._vel_nearest(x,y,12)
            if idx is not None and 0.001<self.vel_nodes[idx][0]<0.999:
                self.vel_nodes.pop(idx); self.vel_area.queue_draw(); self._rerender()
            return
        idx=self._vel_nearest(x,y,12)
        if idx is not None:
            self.vel_drag_idx=idx
        else:
            t=self._vx2t(x); v=self._vy2v(y); self.vel_nodes.append([t,v])
            self.vel_nodes.sort(key=lambda n:n[0])
            self.vel_drag_idx=next(i for i,n in enumerate(self.vel_nodes) if n[0]==t and n[1]==v)
        self.vel_area.queue_draw()

    def _vel_motion(self, widget, event):
        if self.vel_drag_idx is None: return
        x,y=event.x,event.y; t=self._vx2t(x); v=self._vy2v(y)
        nd=self.vel_nodes[self.vel_drag_idx]
        if self.vel_drag_idx==0: t=0.0
        elif self.vel_drag_idx==len(self.vel_nodes)-1: t=1.0
        nd[0]=t; nd[1]=v; old=self.vel_nodes[self.vel_drag_idx]
        self.vel_nodes.sort(key=lambda n:n[0])
        try: self.vel_drag_idx=self.vel_nodes.index(old)
        except ValueError: self.vel_drag_idx=None
        self.vel_area.queue_draw(); self._rerender()

    def _vel_release(self, widget, event): self.vel_drag_idx=None; self.vel_area.queue_draw()
    def _vel_reset(self, w=None):
        self.vel_nodes=[[0.0,1.0],[1.0,1.0]]; self.vel_area.queue_draw(); self._rerender()

    def _vel_curve(self, n):
        nodes=sorted(self.vel_nodes,key=lambda nd:nd[0])
        ts=np.array([nd[0] for nd in nodes]); vs=np.array([nd[1] for nd in nodes])
        return np.interp(np.linspace(0.,1.,n),ts,vs)

    # ══════════════════════════════════════════════════════════════
    #  Odtwarzanie
    # ══════════════════════════════════════════════════════════════
    def on_play(self, w=None):
        if len(self.waveform)<2 or not PYGAME_OK: return
        pygame.mixer.music.load(self._tmp(self.waveform)); pygame.mixer.music.play()
    def on_stop(self, w=None):
        if PYGAME_OK: pygame.mixer.music.stop()
    def _toggle_live(self, w=None):
        self.live_play=not self.live_play; self._st("Live: "+("ON" if self.live_play else "OFF"))
    def _play_now(self):
        if len(self.waveform)<2 or not PYGAME_OK: return
        pygame.mixer.music.load(self._tmp(self.waveform)); pygame.mixer.music.play()
    def _tmp(self, wave):
        p=os.path.join(tempfile.gettempdir(),"ed_waves3.wav")
        wav.write(p,self.sample_rate,(wave*32767).astype(np.int16)); return p

    # ══════════════════════════════════════════════════════════════
    #  Import / Eksport
    # ══════════════════════════════════════════════════════════════
    def on_save_wav(self, widget):
        dlg=Gtk.FileChooserDialog(title="Zapisz WAV",parent=self,action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,Gtk.STOCK_SAVE,Gtk.ResponseType.OK)
        if dlg.run()==Gtk.ResponseType.OK:
            wav.write(dlg.get_filename(),self.sample_rate,(self.waveform*32767).astype(np.int16))
        dlg.destroy()

    def on_import_wav(self, widget):
        dlg=Gtk.FileChooserDialog(title="Wczytaj WAV",parent=self,action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,Gtk.STOCK_OPEN,Gtk.ResponseType.OK)
        if dlg.run()==Gtk.ResponseType.OK:
            sr,data=wav.read(dlg.get_filename())
            if len(data.shape)==2: data=data.mean(axis=1)
            data=data.astype(float); mx=np.max(np.abs(data))
            if mx>0: data/=mx
            self.sample_rate=sr; self._set_wave(data)
        dlg.destroy()

    # ══════════════════════════════════════════════════════════════
    #  Zoom / Scroll / helpers
    # ══════════════════════════════════════════════════════════════
    def _update_view(self):
        if len(self.waveform)==0: return
        total=len(self.waveform); vis=int(total/max(self.zoom_level,0.01))
        self.view_start=max(0,min(self.view_start,total-vis))
        self.view_end=min(total,self.view_start+vis)
        self.vscroll.set_range(0,max(0,total-vis)); self.vscroll.set_value(self.view_start)

    def _on_scroll(self, sb):
        if len(self.waveform)==0: return
        self.view_start=int(sb.get_value())
        vis=int(len(self.waveform)/max(self.zoom_level,0.01))
        self.view_end=min(len(self.waveform),self.view_start+vis)
        self.drawing_area.queue_draw()

    def _crect(self):
        w=self.drawing_area.get_allocated_width()
        h=self.drawing_area.get_allocated_height()
        return FREQ_W,RULER_H,w-FREQ_W,h-RULER_H

    def _st(self, msg): self.status.set_text(msg)


# ═══════════════════════════════════════════════════════════════════
def _sep(): return Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)

if __name__ == "__main__":
    app = WaveformEditor()
    app.show_all()
    Gtk.main()
