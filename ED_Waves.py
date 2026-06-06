"""
ED_Waves v4
===========
Nowości v4:
• SampleStore — format JSON próbki trzymany w pamięci, stabilizuje brzmienie
  między sekcjami (ChaosPad → Edytor i z powrotem).
• MIDIEngine — wybór urządzenia MIDI z GUI, note on/off, pitch bend,
  modulation wheel (CC1), CC7 (volume), CC64 (sustain) reagują na bieżąco.
• Naprawa edytora — zaznaczanie obszaru punktów CP do modyfikacji działa
  poprawnie; ikony narzędzi uproszczone do symboli.
• Suwak "End to" pod obszarem zapętlania — ogranicza odsłuch do zadanego %.
• Typy przenikania między strefami — ostre (hard) lub zachodzące (crossfade)
  jako wybór przy każdym zestawieniu stref.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
import random, tempfile, os, math, json, threading, time

# ── Chaos Pad (opcjonalny) ────────────────────────────────────────
try:
    from chaos_pad import ChaosPadWindow as _ChaosPadWindow
    CHAOS_PAD_OK = True
except ImportError:
    CHAOS_PAD_OK = False

# ── pygame ────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ── python-rtmidi ─────────────────────────────────────────────────
try:
    import rtmidi
    RTMIDI_OK = True
except ImportError:
    RTMIDI_OK = False


# ═══════════════════════════════════════════════════════════════════
#  SampleStore — centralny format próbki (JSON w pamięci)
# ═══════════════════════════════════════════════════════════════════
class SampleStore:
    """
    Trzyma pełny opis próbki jako słownik → JSON.
    Dzięki temu generacja w ChaosPad i w Edytorze używa tych samych
    parametrów i brzmi identycznie przy każdym odtworzeniu.

    Struktura:
    {
      "source": "chaos_pad" | "generator" | "import" | "editor",
      "wtype":  nazwa fali lub "chaos_pad",
      "duration": float,            # sekundy
      "sr": int,                    # sample rate
      "params": {...},              # parametry syntezatora (chaos_pad / generator)
      "wave_b64": str | None,       # base64-encoded float32 LE (opcjonalny cache)
      "cps": [[idx, amp], ...],     # punkty kontrolne
      "zones": [...],               # strefy loopowania
      "vel_nodes": [[t, v], ...],   # krzywa velocity
      "created_at": float,          # timestamp
    }
    """
    VERSION = 4

    def __init__(self):
        self._store: dict = self._blank()

    def _blank(self) -> dict:
        return {
            "version": self.VERSION,
            "source": "none",
            "wtype": "sine",
            "duration": 1.0,
            "sr": 44100,
            "params": {},
            "cps": [],
            "zones": [],
            "vel_nodes": [[0.0, 1.0], [1.0, 1.0]],
            "created_at": time.time(),
        }

    # ── gettery / settery ─────────────────────────────────────────
    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value
        self._store["modified_at"] = time.time()

    def update_from_wave(self, wave: np.ndarray, cps: list,
                         source: str, sr: int = 44100, **kwargs):
        """Aktualizuje store po zmianie fali."""
        self._store.update({
            "source": source,
            "sr": sr,
            "cps": [list(c) for c in cps],
            "created_at": time.time(),
            **kwargs
        })

    def update_cps(self, cps: list):
        self._store["cps"] = [list(c) for c in cps]

    def update_zones(self, zones: list):
        self._store["zones"] = [dict(z) for z in zones]

    def update_vel(self, vel_nodes: list):
        self._store["vel_nodes"] = [list(n) for n in vel_nodes]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._store, indent=indent, ensure_ascii=False)

    def from_json(self, text: str):
        self._store = json.loads(text)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.from_json(f.read())

    def summary(self) -> str:
        src = self._store.get("source","?")
        wt  = self._store.get("wtype","?")
        dur = self._store.get("duration", 0)
        cps = len(self._store.get("cps", []))
        return f"[{src}] {wt} {dur:.2f}s  CP={cps}"


# ═══════════════════════════════════════════════════════════════════
#  MIDIEngine — obsługa urządzenia MIDI
# ═══════════════════════════════════════════════════════════════════
class MIDIEngine:
    """
    Otwiera wybrany port MIDI i wywołuje callbacki:
      on_note_on(note, velocity)
      on_note_off(note)
      on_pitchbend(value)   # -1.0..+1.0
      on_cc(cc, value)      # 0..127
    """
    def __init__(self):
        self._midiin = None
        self._port_idx = -1
        self.on_note_on   = None
        self.on_note_off  = None
        self.on_pitchbend = None
        self.on_cc        = None
        self._sustain     = False

    def list_ports(self) -> list:
        if not RTMIDI_OK:
            return []
        try:
            m = rtmidi.MidiIn()
            ports = m.get_ports()
            del m
            return ports
        except Exception:
            return []

    def open(self, idx: int) -> bool:
        if not RTMIDI_OK:
            return False
        self.close()
        try:
            self._midiin = rtmidi.MidiIn()
            self._midiin.open_port(idx)
            self._midiin.set_callback(self._callback)
            self._midiin.ignore_types(sysex=True, timing=True, active_sense=True)
            self._port_idx = idx
            return True
        except Exception as e:
            print(f"MIDI open error: {e}")
            self._midiin = None
            return False

    def close(self):
        if self._midiin:
            try:
                self._midiin.close_port()
            except Exception:
                pass
            self._midiin = None
        self._port_idx = -1

    def _callback(self, message, data=None):
        msg, dt = message
        if not msg:
            return
        status = msg[0] & 0xF0
        ch     = msg[0] & 0x0F

        if status == 0x90:          # Note On
            note = msg[1]; vel = msg[2]
            if vel == 0:
                if self.on_note_off: self.on_note_off(note)
            else:
                if self.on_note_on:  self.on_note_on(note, vel)

        elif status == 0x80:        # Note Off
            note = msg[1]
            if self.on_note_off: self.on_note_off(note)

        elif status == 0xB0:        # Control Change
            cc = msg[1]; val = msg[2]
            if self.on_cc: self.on_cc(cc, val)

        elif status == 0xE0:        # Pitch Bend
            lsb = msg[1]; msb = msg[2]
            raw = (msb << 7) | lsb   # 0..16383, środek=8192
            norm = (raw - 8192) / 8192.0
            if self.on_pitchbend: self.on_pitchbend(norm)


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

def _lead_c(cps,n):
    env=_adsr(n,0.005,0.05,0.8,0.1,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("lead", _lead_c, lambda w,sr: _normalize(_soft(w,1.4)))

def _pad_c(cps,n):
    env=_adsr(n,0.3,0.1,0.9,0.5,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("pad", _pad_c, lambda w,sr: _normalize(_lp(w,4000,sr,2)))

def _bass_c(cps,n):
    env=_adsr(n,0.01,0.08,0.85,0.15,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("bass", _bass_c, lambda w,sr: _normalize(_soft(_lp(w,2500,sr,2),1.8)))

def _pluck_c(cps,n):
    env=_adsr(n,0.002,0.25,0.0,0.05,SR)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("pluck", _pluck_c, lambda w,sr: _normalize(_hp(w,80,sr)))

def _arp_c(cps,n):
    gate=np.ones(n); gate[n//2:]=0.0
    for cp in cps: cp[1]*=gate[min(cp[0],n-1)]
_reg("arp", _arp_c, None)

def _kick_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*18)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("kick", _kick_c, lambda w,sr: _normalize(_soft(_lp(w,120,sr,2)*1.5,2.0)))

def _snare_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*30)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
def _snare_d(w,sr):
    noise=np.random.normal(0,0.3,len(w))*np.exp(-np.arange(len(w))/sr*25)
    return _normalize(_hp(w+noise,200,sr))
_reg("snare", _snare_c, _snare_d)

def _hat_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*80)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("hat", _hat_c, lambda w,sr: _normalize(_hp(w,6000,sr,2)))

def _tom_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*12)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
_reg("tom", _tom_c, lambda w,sr: _normalize(_lp(_soft(w,1.2),800,sr,2)))

def _clap_c(cps,n):
    t=np.arange(n)/SR; env=np.exp(-t*40)
    for cp in cps: cp[1]*=env[min(cp[0],n-1)]
def _clap_d(w,sr):
    noise=np.random.normal(0,1.0,len(w))*np.exp(-np.arange(len(w))/sr*40)
    return _normalize(_hp(w*0.3+noise*0.7,500,sr))
_reg("clap", _clap_c, _clap_d)

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
#  Dialog wyboru urządzenia MIDI
# ═══════════════════════════════════════════════════════════════════
class MIDIDeviceDialog(Gtk.Dialog):
    def __init__(self, parent, ports):
        super().__init__(title="Wybór urządzenia MIDI", parent=parent,
                         flags=Gtk.DialogFlags.MODAL)
        self.set_default_size(420, 280)
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_CONNECT, Gtk.ResponseType.OK
        )
        self.selected_idx = -1

        box = self.get_content_area()
        box.set_spacing(8); box.set_border_width(10)

        if not RTMIDI_OK:
            box.add(Gtk.Label(label="Brak python-rtmidi.\nZainstaluj: pip install python-rtmidi"))
            return

        if not ports:
            box.add(Gtk.Label(label="Nie znaleziono urządzeń MIDI."))
            return

        box.add(Gtk.Label(label="Dostępne porty MIDI:"))
        self.liststore = Gtk.ListStore(int, str)
        for i, p in enumerate(ports):
            self.liststore.append([i, p])

        tv = Gtk.TreeView(model=self.liststore)
        tv.set_headers_visible(True)
        tv.append_column(Gtk.TreeViewColumn("#",    Gtk.CellRendererText(), text=0))
        tv.append_column(Gtk.TreeViewColumn("Port", Gtk.CellRendererText(), text=1))
        tv.connect("row-activated", self._row_activated)

        sel = tv.get_selection()
        sel.set_mode(Gtk.SelectionMode.SINGLE)
        sel.connect("changed", self._sel_changed)
        if len(ports) > 0:
            sel.select_path(Gtk.TreePath.new_first())

        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        sw.set_size_request(-1, 160)
        sw.add(tv)
        box.add(sw)
        box.show_all()

    def _sel_changed(self, sel):
        model, it = sel.get_selected()
        if it:
            self.selected_idx = model[it][0]

    def _row_activated(self, tv, path, col):
        self.response(Gtk.ResponseType.OK)


# ═══════════════════════════════════════════════════════════════════
#  Główna klasa WaveformEditor
# ═══════════════════════════════════════════════════════════════════
class WaveformEditor(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="ED_Waves v4")
        self.set_default_size(1440, 900)
        self.connect("destroy", self._on_destroy)

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
        # CP zaznaczone przez rect-select
        self.sel_cp_set    = set()

        self.current_tool  = TOOL_DRAW
        self.current_line  = LINE_BEZIER
        self.current_corr  = CORR_MUL
        self.show_cps      = True
        self.sample_w_px   = 2

        self.pre_profile   = None
        self.post_profile  = None
        self.post_dsp_name = None   # non-destructive: nazwa profilu post-DSP

        self.vel_nodes     = [[0.0,1.0],[1.0,1.0]]
        self.vel_drag_idx  = None
        self.vel_apply     = True
        self.live_play     = False

        # strefy loopowania
        self.zones          = []
        self.zone_drag      = None
        self.zone_sel_idx   = None
        self.zone_mode      = 'sequential'
        self.zone_rendered  = np.array([])
        # Typ przenikania między strefami (globalny domyślny)
        self.zone_crossfade = 'sharp'   # 'sharp' lub 'crossfade'
        # End-to suwak (0.0..1.0)
        self.play_end_frac  = 1.0

        # SampleStore
        self.store = SampleStore()

        # MIDI
        self.midi = MIDIEngine()
        self.midi.on_note_on   = self._midi_note_on
        self.midi.on_note_off  = self._midi_note_off
        self.midi.on_pitchbend = self._midi_pitchbend
        self.midi.on_cc        = self._midi_cc
        self._midi_pitch_bend  = 0.0  # -1..+1
        self._midi_modulation  = 0.0  # 0..1
        self._midi_volume      = 1.0  # 0..1
        self._midi_sustain     = False
        self._midi_playing_note = None

        self._chaos_pad_win = None
        self._build_ui()

    def _on_destroy(self, widget):
        self.midi.close()
        Gtk.main_quit()

    # ════════════════════════ UI ══════════════════════════════════
    def _build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add(root)

        # ── pasek 1: narzędzia ────────────────────────────────────
        tb1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        tb1.set_border_width(3); root.pack_start(tb1,False,False,0)

        # Narzędzia — uproszczone ikony symboliczne
        tg=None
        for lbl,t in [("✏",TOOL_DRAW),("⊙",TOOL_EDIT),("⬚",TOOL_SELECT)]:
            b=Gtk.RadioButton.new_with_label_from_widget(tg,lbl)
            if tg is None: tg=b
            b.set_tooltip_text({"✏":"Rysuj warstwę korekcji",
                                "⊙":"Edytuj punkt",
                                "⬚":"Zaznacz obszar"}[lbl])
            b.connect("toggled",self._on_tool,t)
            tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        lg=None
        for lbl,lt,tip in [("~",LINE_BEZIER,"Bézier"),
                            ("∕",LINE_LINEAR,"Linia prosta"),
                            ("⌐",LINE_STEP,  "Schodek")]:
            b=Gtk.RadioButton.new_with_label_from_widget(lg,lbl)
            if lg is None: lg=b
            b.set_tooltip_text(tip)
            b.connect("toggled",self._on_linetype,lt)
            tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        cg=None
        for lbl,cm,tip in [("×",CORR_MUL,"MUL: mnożnik amplitudy"),
                            ("±",CORR_ADD,"ADD: dodawanie amplitudy")]:
            b=Gtk.RadioButton.new_with_label_from_widget(cg,lbl)
            if cg is None: cg=b
            b.set_tooltip_text(tip)
            b.connect("toggled",self._on_corrmode,cm)
            tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        for lbl,cb,tip in [("▶","on_play","Play waveform"),
                            ("⏹","on_stop","Stop"),
                            ("⟳","_toggle_live","Live re-play"),
                            ("💾","on_save_wav","Zapisz WAV"),
                            ("📂","on_import_wav","Wczytaj WAV"),
                            ("📋","_save_json","Zapisz JSON próbki"),
                            ("🗑","_clear_layers","Wyczyść warstwy")]:
            b=Gtk.Button(label=lbl)
            b.set_tooltip_text(tip)
            b.connect("clicked",getattr(self,cb))
            tb1.pack_start(b,False,False,0)

        tb1.pack_start(_sep(),False,False,4)
        tb1.pack_start(Gtk.Label(label="px/prb:"),False,False,0)
        self.sw_spin=Gtk.SpinButton()
        self.sw_spin.set_range(1,20); self.sw_spin.set_value(2)
        self.sw_spin.set_increments(1,2)
        self.sw_spin.connect("value-changed",lambda s:setattr(self,'sample_w_px',int(s.get_value())))
        tb1.pack_start(self.sw_spin,False,False,0)

        chk=Gtk.CheckButton(label="CP"); chk.set_active(True)
        chk.set_tooltip_text("Pokaż punkty kontrolne")
        chk.connect("toggled",lambda w:[setattr(self,'show_cps',w.get_active()),
                                        self.drawing_area.queue_draw()])
        tb1.pack_start(chk,False,False,0)

        # MIDI button
        tb1.pack_start(_sep(),False,False,4)
        midi_btn = Gtk.Button(label="🎹 MIDI")
        midi_btn.set_tooltip_text("Wybierz urządzenie MIDI")
        midi_btn.connect("clicked", self._open_midi_dialog)
        tb1.pack_start(midi_btn, False,False,0)
        self.midi_status_lbl = Gtk.Label(label="⬤ OFF")
        self.midi_status_lbl.set_markup('<span foreground="red">⬤ OFF</span>')
        tb1.pack_start(self.midi_status_lbl, False,False,2)

        # MIDI indicators
        self.midi_pb_lbl  = Gtk.Label(label="PB:  0.00")
        self.midi_mod_lbl = Gtk.Label(label="MOD:0.00")
        self.midi_note_lbl= Gtk.Label(label="NOTE:---")
        for lbl in [self.midi_pb_lbl, self.midi_mod_lbl, self.midi_note_lbl]:
            lbl.set_width_chars(10)
            tb1.pack_start(lbl, False,False,0)

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
        tb2.pack_start(Gtk.Label(label="Pre:"),False,False,0)
        self.pre_combo=Gtk.ComboBoxText(); self.pre_combo.append_text("(brak)")
        for p in SoundProfile.names(): self.pre_combo.append_text(p)
        self.pre_combo.set_active(0); tb2.pack_start(self.pre_combo,False,False,0)
        bp=Gtk.Button(label="▶"); bp.set_tooltip_text("Zastosuj Pre-profil")
        bp.connect("clicked",self._apply_pre); tb2.pack_start(bp,False,False,0)

        tb2.pack_start(_sep(),False,False,4)
        tb2.pack_start(Gtk.Label(label="Post:"),False,False,0)
        self.post_combo=Gtk.ComboBoxText(); self.post_combo.append_text("(brak)")
        for p in SoundProfile.names(): self.post_combo.append_text(p)
        self.post_combo.set_active(0); tb2.pack_start(self.post_combo,False,False,0)
        bp2=Gtk.Button(label="▶"); bp2.set_tooltip_text("Zastosuj Post-profil DSP")
        bp2.connect("clicked",self._apply_post); tb2.pack_start(bp2,False,False,0)

        br=Gtk.Button(label="↩"); br.set_tooltip_text("Reset do oryginału")
        br.connect("clicked",self._reset_all); tb2.pack_start(br,False,False,0)

        tb2.pack_start(_sep(),False,False,6)
        bcp=Gtk.Button(label="🎛 Chaos Pad")
        bcp.connect("clicked",self._open_chaos_pad)
        tb2.pack_start(bcp,False,False,0)

        # Store info label
        self.store_lbl = Gtk.Label(label="")
        self.store_lbl.set_xalign(0)
        tb2.pack_start(self.store_lbl, False,False,8)

        # ── pasek 3: transformacje CP ─────────────────────────────
        tb3=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=4)
        tb3.set_border_width(2); root.pack_start(tb3,False,False,0)
        tb3.pack_start(Gtk.Label(label="Zaznaczone CP:"),False,False,0)
        # Uproszczone ikony dla transformacji
        for lbl,act,tip in [("▼Y",'compress_y',"Ściskaj Y"),
                             ("▲Y",'stretch_y', "Rozciągaj Y"),
                             ("◄X",'compress_x',"Ściskaj X"),
                             ("►X",'stretch_x', "Rozciągaj X"),
                             ("⇅", 'flip_y',    "Odwróć Y"),
                             ("⇄", 'flip_x',    "Odwróć X"),
                             ("●∘",'toggle',    "Wł/Wył warstwę"),
                             ("✕", 'delete',    "Usuń warstwę")]:
            b=Gtk.Button(label=lbl)
            b.set_tooltip_text(tip)
            b.connect("clicked",lambda w,a=act:self._transform_sel(a))
            tb3.pack_start(b,False,False,0)
        tb3.pack_start(Gtk.Label(label=" sY:"),False,False,0)
        self.sel_sy=Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_sy.set_range(0.05,4.0); self.sel_sy.set_value(1.0)
        self.sel_sy.set_digits(2); self.sel_sy.set_size_request(80,-1)
        tb3.pack_start(self.sel_sy,False,False,0)
        tb3.pack_start(Gtk.Label(label=" sX:"),False,False,0)
        self.sel_sx=Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_sx.set_range(0.05,4.0); self.sel_sx.set_value(1.0)
        self.sel_sx.set_digits(2); self.sel_sx.set_size_request(80,-1)
        tb3.pack_start(self.sel_sx,False,False,0)

        # ── drawing area ──────────────────────────────────────────
        hb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        root.pack_start(hb,True,True,0)
        self.drawing_area=Gtk.DrawingArea()
        self.drawing_area.set_size_request(1280,400)
        self.drawing_area.connect("draw",self.on_draw)
        self.drawing_area.connect("button-press-event",  self.on_press)
        self.drawing_area.connect("button-release-event",self.on_release)
        self.drawing_area.connect("motion-notify-event", self.on_motion)
        mask=(Gdk.EventMask.BUTTON_PRESS_MASK|Gdk.EventMask.BUTTON_RELEASE_MASK|
              Gdk.EventMask.POINTER_MOTION_MASK)
        self.drawing_area.set_events(mask)
        hb.pack_start(self.drawing_area,True,True,0)
        self.vscroll=Gtk.VScrollbar(); self.vscroll.connect("value-changed",self._on_scroll)
        hb.pack_start(self.vscroll,False,False,0)

        # ── velocity ──────────────────────────────────────────────
        vf=Gtk.Frame(label=" Velocity (LPM=węzeł, PPM=usuń) ")
        vf.set_border_width(2); root.pack_start(vf,False,False,0)
        vvb=Gtk.Box(orientation=Gtk.Orientation.VERTICAL); vf.add(vvb)
        vtb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=5)
        vtb.set_border_width(2); vvb.pack_start(vtb,False,False,0)
        vchk=Gtk.CheckButton(label="Velocity"); vchk.set_active(True)
        vchk.connect("toggled",lambda w:[setattr(self,'vel_apply',w.get_active()),self._rerender()])
        vtb.pack_start(vchk,False,False,0)
        vrst=Gtk.Button(label="Reset 100%"); vrst.connect("clicked",self._vel_reset)
        vtb.pack_start(vrst,False,False,0)
        self.vel_area=Gtk.DrawingArea(); self.vel_area.set_size_request(-1,65)
        self.vel_area.connect("draw",self._vel_draw)
        self.vel_area.connect("button-press-event",  self._vel_press)
        self.vel_area.connect("button-release-event",self._vel_release)
        self.vel_area.connect("motion-notify-event", self._vel_motion)
        self.vel_area.set_events(mask); vvb.pack_start(self.vel_area,True,True,0)

        # ── strefy loopowania ─────────────────────────────────────
        zf = Gtk.Frame(label=" Strefy loopowania (LPM=nowa, PPM=usuń, drag=granica) ")
        zf.set_border_width(2); root.pack_start(zf, False, False, 0)
        zvb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL); zf.add(zvb)

        ztb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        ztb.set_border_width(2); zvb.pack_start(ztb, False, False, 0)

        ztb.pack_start(Gtk.Label(label="Tryb:"), False, False, 0)
        self.zone_mode_combo = Gtk.ComboBoxText()
        for zm in ["sequential","layer","nested"]:
            self.zone_mode_combo.append_text(zm)
        self.zone_mode_combo.set_active(0)
        self.zone_mode_combo.connect("changed", lambda w: setattr(self,'zone_mode',w.get_active_text()))
        ztb.pack_start(self.zone_mode_combo, False, False, 0)

        # Typ przenikania — NOWE
        ztb.pack_start(Gtk.Label(label=" Przenikanie:"), False, False, 0)
        self.xfade_combo = Gtk.ComboBoxText()
        self.xfade_combo.append_text("ostre")
        self.xfade_combo.append_text("zachodzące")
        self.xfade_combo.set_active(0)
        self.xfade_combo.set_tooltip_text("Typ przejścia między sekcjami stref")
        self.xfade_combo.connect("changed", self._xfade_changed)
        ztb.pack_start(self.xfade_combo, False, False, 0)

        for lbl,cb in [("▶ Play", self._zone_play),
                       ("⏹ Stop", self._zone_stop),
                       ("📋→Fala",self._zone_render_to_wave),
                       ("🗑",     self._zone_clear)]:
            b=Gtk.Button(label=lbl); b.connect("clicked",cb)
            ztb.pack_start(b,False,False,0)

        ztb.pack_start(Gtk.Label(label="  Z zaznaczona:"), False, False, 0)
        self.zone_repeats_spin = Gtk.SpinButton()
        self.zone_repeats_spin.set_range(1,32); self.zone_repeats_spin.set_value(1)
        self.zone_repeats_spin.set_increments(1,4)
        self.zone_repeats_spin.connect("value-changed", self._zone_repeats_changed)
        ztb.pack_start(Gtk.Label(label="×"), False, False, 0)
        ztb.pack_start(self.zone_repeats_spin, False, False, 0)

        b_add_pass = Gtk.Button(label="+ Przebieg")
        b_add_pass.connect("clicked", self._zone_add_pass)
        ztb.pack_start(b_add_pass, False, False, 0)
        b_del_pass = Gtk.Button(label="− Przebieg")
        b_del_pass.connect("clicked", self._zone_del_pass)
        ztb.pack_start(b_del_pass, False, False, 0)

        # Obszar stref
        self.zone_area = Gtk.DrawingArea()
        self.zone_area.set_size_request(-1, 90)
        self.zone_area.connect("draw", self._zone_draw)
        self.zone_area.connect("button-press-event",   self._zone_press)
        self.zone_area.connect("button-release-event", self._zone_release)
        self.zone_area.connect("motion-notify-event",  self._zone_motion)
        self.zone_area.set_events(mask)
        zvb.pack_start(self.zone_area, False, False, 0)

        # ── Suwak End-to (nowy!) ──────────────────────────────────
        etb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        etb.set_border_width(3)
        zvb.pack_start(etb, False, False, 0)
        etb.pack_start(Gtk.Label(label="End to:"), False, False, 0)
        self.end_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.end_scale.set_range(0.05, 1.0); self.end_scale.set_value(1.0)
        self.end_scale.set_digits(2)
        self.end_scale.set_hexpand(True)
        self.end_scale.set_draw_value(True)
        self.end_scale.connect("value-changed", self._end_scale_changed)
        etb.pack_start(self.end_scale, True, True, 0)
        self.end_lbl = Gtk.Label(label="100%")
        self.end_lbl.set_width_chars(6)
        etb.pack_start(self.end_lbl, False, False, 0)

        # Panel przebiegów
        self.pass_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.pass_box.set_border_width(2)
        zvb.pack_start(self.pass_box, False, False, 0)

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
        root.pack_start(self.status,False,False,0)

    # ══════════════════════════════════════════════════════════════
    #  End-to suwak
    # ══════════════════════════════════════════════════════════════
    def _end_scale_changed(self, widget):
        self.play_end_frac = widget.get_value()
        pct = int(self.play_end_frac * 100)
        self.end_lbl.set_text(f"{pct}%")
        self.zone_area.queue_draw()

    def _wave_trimmed(self) -> np.ndarray:
        """Zwraca finalną falę (non-destructive render) obciętą do play_end_frac."""
        wave = self._build_render_wave()
        if len(wave) < 2:
            return wave
        end = max(2, int(len(wave) * self.play_end_frac))
        return wave[:end]

    # ══════════════════════════════════════════════════════════════
    #  Crossfade wybór
    # ══════════════════════════════════════════════════════════════
    def _xfade_changed(self, combo):
        mapping = {"ostre": "sharp", "zachodzące": "crossfade"}
        self.zone_crossfade = mapping.get(combo.get_active_text(), "sharp")

    # ══════════════════════════════════════════════════════════════
    #  MIDI callbacks
    # ══════════════════════════════════════════════════════════════
    def _midi_note_on(self, note, velocity):
        self._midi_playing_note = note
        vel_norm = velocity / 127.0
        GLib.idle_add(self._midi_update_ui)
        # Play waveform at this note pitch shifted
        if len(self.waveform) > 1 and PYGAME_OK:
            wave = self._wave_trimmed()
            # pitch-shift via resample ratio
            base_note = 69  # A4=440Hz
            ratio = 2.0 ** ((note - base_note) / 12.0)
            # velocity scaling
            wave_out = wave * vel_norm
            # apply pitch by resampling
            n_new = max(2, int(len(wave_out) / ratio))
            wave_pitched = np.interp(
                np.linspace(0, len(wave_out)-1, n_new),
                np.arange(len(wave_out)),
                wave_out
            )
            self._play_wave(wave_pitched)

    def _midi_note_off(self, note):
        if self._midi_playing_note == note:
            self._midi_playing_note = None
            if not self._midi_sustain:
                GLib.idle_add(self._do_midi_stop)
        GLib.idle_add(self._midi_update_ui)

    def _do_midi_stop(self):
        if PYGAME_OK and not self._midi_sustain and self._midi_playing_note is None:
            pygame.mixer.music.stop()

    def _midi_pitchbend(self, value):
        self._midi_pitch_bend = value
        GLib.idle_add(self._midi_update_ui)

    def _midi_cc(self, cc, val):
        norm = val / 127.0
        if cc == 1:     # Modulation wheel
            self._midi_modulation = norm
        elif cc == 7:   # Volume
            self._midi_volume = norm
        elif cc == 64:  # Sustain pedal
            self._midi_sustain = val >= 64
            if not self._midi_sustain and self._midi_playing_note is None:
                GLib.idle_add(self._do_midi_stop)
        GLib.idle_add(self._midi_update_ui)

    def _midi_update_ui(self):
        self.midi_pb_lbl.set_text(f"PB:{self._midi_pitch_bend:+.2f}")
        self.midi_mod_lbl.set_text(f"MOD:{self._midi_modulation:.2f}")
        note = self._midi_playing_note
        if note is not None:
            names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            nm = names[note % 12] + str(note // 12 - 1)
            self.midi_note_lbl.set_text(f"♪ {nm}")
        else:
            self.midi_note_lbl.set_text("NOTE:---")

    def _open_midi_dialog(self, widget=None):
        ports = self.midi.list_ports()
        dlg = MIDIDeviceDialog(self, ports)
        resp = dlg.run()
        if resp == Gtk.ResponseType.OK and dlg.selected_idx >= 0:
            if self.midi.open(dlg.selected_idx):
                port_name = ports[dlg.selected_idx] if ports else "?"
                self.midi_status_lbl.set_markup(
                    f'<span foreground="lime">⬤ {port_name[:18]}</span>')
                self._st(f"MIDI: {port_name}")
            else:
                self.midi_status_lbl.set_markup('<span foreground="red">⬤ BŁĄD</span>')
        dlg.destroy()

    # ══════════════════════════════════════════════════════════════
    #  Callbacki narzędzi
    # ══════════════════════════════════════════════════════════════
    def _on_tool(self,btn,t):
        if btn.get_active():
            self.current_tool=t
            # czyść zaznaczenie CP przy zmianie narzędzia
            if t != TOOL_SELECT:
                self.sel_cp_set.clear()
                self.drawing_area.queue_draw()

    def _on_linetype(self,btn,lt):
        if btn.get_active():
            self.current_line=lt
            for l in self.layers:
                if l.selected: l.line_type=lt
            self.drawing_area.queue_draw()

    def _on_corrmode(self,btn,cm):
        if btn.get_active(): self.current_corr=cm

    # ══════════════════════════════════════════════════════════════
    #  SampleStore — JSON
    # ══════════════════════════════════════════════════════════════
    def _update_store(self, source="editor", extra=None):
        extra = extra or {}
        self.store.update_from_wave(
            self.original_wave, self.wave_cps,
            source=source, sr=self.sample_rate,
            duration=extra.pop("duration", len(self.original_wave)/max(self.sample_rate,1)),
            wtype=extra.pop("wtype", self.gen_combo.get_active_text() or "custom"),
            **extra
        )
        self.store.update_zones(self.zones)
        self.store.update_vel(self.vel_nodes)
        self.store_lbl.set_text(self.store.summary())

    def _save_json(self, widget=None):
        dlg=Gtk.FileChooserDialog(title="Zapisz próbkę JSON",parent=self,
                                  action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_SAVE,  Gtk.ResponseType.OK)
        filt=Gtk.FileFilter(); filt.set_name("JSON (*.json)"); filt.add_pattern("*.json")
        dlg.add_filter(filt)
        if dlg.run()==Gtk.ResponseType.OK:
            path=dlg.get_filename()
            if not path.endswith(".json"): path+=".json"
            self._update_store()
            self.store.save(path)
            self._st(f"Zapisano JSON: {path}")
        dlg.destroy()

    # ══════════════════════════════════════════════════════════════
    #  Generator
    # ══════════════════════════════════════════════════════════════
    def _on_generate(self, widget):
        wtype = self.gen_combo.get_active_text() or "sine"
        dur   = self.gen_dur.get_value()
        wave  = generate_base_wave(wtype, dur, self.sample_rate)
        self._set_wave(wave, source="generator", wtype=wtype, duration=dur)
        self._st(f"Wygenerowano: {wtype}  {dur:.2f}s  {len(wave)} próbek  CP={len(self.wave_cps)}")

    def _set_wave(self, wave, source="editor", **kwargs):
        self.original_wave = wave.copy()
        self.post_dsp_name = None          # nowa fala — reset post-DSP
        self.wave_cps      = extract_control_points(wave)
        self.layers        = []
        self.sel_cp_set    = set()
        self.view_start    = 0; self.view_end = len(wave)
        self._rerender(); self._update_view()
        self.drawing_area.queue_draw(); self.vel_area.queue_draw()
        self._update_store(source=source, extra=kwargs)

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
        self.original_wave=new_wave
        self.wave_cps=extract_control_points(new_wave)
        self._rerender(); self.drawing_area.queue_draw()
        self._update_store(source="pre_profile", extra={"pre_profile": name})
        self._st(f"Pre-profil: {name}  CP={len(self.wave_cps)}")

    def _apply_post(self, widget):
        name=self._pname(self.post_combo)
        if not name or len(self.original_wave)<2: return
        self.post_dsp_name = name   # non-destructive: aplikowane przy każdym render
        self.drawing_area.queue_draw()
        if self.live_play: self._play_now()
        self._st(f"Post DSP: {name}  (non-destructive)")

    def _open_chaos_pad(self, widget=None):
        if not CHAOS_PAD_OK:
            dlg=Gtk.MessageDialog(parent=self,flags=0,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Brak pliku chaos_pad.py w tym samym katalogu co ED_Waves.py")
            dlg.run(); dlg.destroy(); return
        if self._chaos_pad_win is None or not self._chaos_pad_win.get_visible():
            self._chaos_pad_win = _ChaosPadWindow(callback=self._chaos_pad_callback)
            self._chaos_pad_win.show_all()
        else:
            self._chaos_pad_win.present()

    def _chaos_pad_callback(self, wave, params):
        """Odbiera falę z Chaos Pada + zapisuje parametry w SampleStore."""
        if wave is None or len(wave) < 4:
            return
        mx = np.max(np.abs(wave))
        if mx > 1e-9: wave = wave / mx
        self.original_wave = wave.copy()
        self.base_wave     = wave.copy()   # czyste źródło bez CP-interpolacji
        self.waveform      = wave.copy()   # widok (do rysowania)
        self.post_dsp_name = None          # reset post-DSP przy nowej fali
        self.wave_cps      = extract_control_points(wave)
        self.layers        = []
        self.sel_cp_set    = set()
        self.view_start    = 0; self.view_end = len(wave)
        self._update_view()
        GLib.idle_add(self.drawing_area.queue_draw)
        GLib.idle_add(self.vel_area.queue_draw)
        f0   = params.get('f0', 0)
        poly = params.get('poly', 1)
        # Zapisz parametry ChaosPad w SampleStore — gwarantuje stabilność brzmienia
        self._update_store(
            source="chaos_pad",
            extra={
                "wtype": "chaos_pad",
                "duration": len(wave)/self.sample_rate,
                "params": {k: (v if not isinstance(v, (list,np.ndarray))
                               else [float(x) for x in v])
                           for k,v in params.items()},
            }
        )
        self._st(f"Chaos Pad → f0={f0:.1f}Hz  poly={poly:.1f}  CP={len(self.wave_cps)}")

    def _reset_all(self, widget):
        self.wave_cps=extract_control_points(self.original_wave)
        self.layers=[]; self.sel_cp_set=set()
        self._rerender(); self.drawing_area.queue_draw()
        self._update_store(source="reset")
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
        # base_wave = CP-rekonstrukcja bez vel i post-DSP (czyste źródło)
        self.base_wave = wave
        # waveform = base_wave (do rysowania — bez time-warp żeby nie zmieniać długości widoku)
        self.waveform=wave
        if self.live_play: self._play_now()
        # Uaktualnij CP w store
        self.store.update_cps(cps)

    def _build_render_wave(self):
        """Buduje falę do odtwarzania non-destructive:
        original_wave → CP layers → base_wave → vel_curve → post_dsp
        Nigdy nie mutuje żadnego pola stanu."""
        if len(self.original_wave) < 2:
            return self.original_wave.copy()
        # Użyj base_wave jeśli dostępne (obliczone przez _rerender),
        # wpp. zrekonstruuj bezpośrednio z original_wave
        if hasattr(self, 'base_wave') and len(self.base_wave) == len(self.original_wave):
            wave = self.base_wave.copy()
        else:
            wave = self.waveform.copy() if len(self.waveform) > 1 else self.original_wave.copy()
        # Aplikuj krzywą prędkości (time-warp)
        if self.vel_apply:
            wave = self._apply_speed_warp(wave)
        # Aplikuj post-DSP (bez mutowania waveform)
        if self.post_dsp_name:
            wave = SoundProfile.apply_dsp(self.post_dsp_name, wave, self.sample_rate)
            mx = np.max(np.abs(wave))
            if mx > 1e-9: wave = wave / mx
        return wave

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
        # oś środkowa
        cr.set_source_rgba(0.5,0.5,0.5,0.6); cr.set_line_width(1)
        cr.move_to(ox,oy+ch/2); cr.line_to(ox+cw,oy+ch/2); cr.stroke()
        # waveform
        if len(self.waveform)>1: self._draw_wave(cr,ox,oy,cw,ch)
        # CP — zaznaczone/niezaznaczone
        if self.show_cps and self.wave_cps:
            self._draw_wave_cps(cr,ox,oy,cw,ch)
        # Warstwy korekcji
        for layer in self.layers: layer.draw(cr)
        # Podgląd rysowania nowej warstwy
        if self.draw_start and self.current_tool==TOOL_DRAW:
            cr.set_source_rgba(0.3,0.9,1.0,0.45); cr.set_line_width(1.5); cr.set_dash([6,4])
            cr.move_to(*self.draw_start); cr.line_to(*self.mouse_pos); cr.stroke()
            cr.set_dash([]); cr.set_source_rgba(0.3,0.9,1.0,0.85)
            cr.arc(self.draw_start[0],self.draw_start[1],CP_R+1,0,6.28); cr.fill()
        # Prostokąt zaznaczenia
        if self.sel_rect_s and self.sel_rect_e:
            x1,y1=self.sel_rect_s; x2,y2=self.sel_rect_e
            rx,ry=min(x1,x2),min(y1,y2); rw,rh=abs(x2-x1),abs(y2-y1)
            cr.set_source_rgba(0.3,0.75,1.0,0.12); cr.rectangle(rx,ry,rw,rh); cr.fill()
            cr.set_source_rgba(0.3,0.75,1.0,0.75); cr.set_line_width(1.5); cr.set_dash([4,3])
            cr.rectangle(rx,ry,rw,rh); cr.stroke(); cr.set_dash([])
        # End-to marker
        if len(self.waveform)>1:
            end_x = ox + self.play_end_frac * cw
            cr.set_source_rgba(1.0, 0.3, 0.3, 0.8)
            cr.set_line_width(1.5); cr.set_dash([5,3])
            cr.move_to(end_x, oy); cr.line_to(end_x, oy+ch); cr.stroke()
            cr.set_dash([])
            cr.set_font_size(8); cr.move_to(end_x+2, oy+10)
            cr.show_text(f"END {int(self.play_end_frac*100)}%")

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
        """Rysuje CP — zaznaczone świecą na biało/żółto, niezaznaczone pomarańczowo."""
        n=len(self.original_wave); vlen=max(1,self.view_end-self.view_start)
        for i, cp in enumerate(self.wave_cps):
            idx,amp=cp[0],cp[1]
            t_view=(idx-self.view_start)/vlen
            if t_view<-0.01 or t_view>1.01: continue
            x=ox+t_view*cw; y=oy+ch/2-amp*ch/2
            is_sel = i in self.sel_cp_set
            if is_sel:
                cr.set_source_rgba(1.0, 0.92, 0.15, 0.95)
                cr.arc(x, y, 5, 0, 6.28); cr.fill()
                cr.set_source_rgba(1,1,1,0.8)
                cr.arc(x, y, 5, 0, 6.28)
                cr.set_line_width(1.5); cr.stroke()
            else:
                cr.set_source_rgba(0.9, 0.5, 0.1, 0.70)
                cr.arc(x, y, 3, 0, 6.28); cr.fill()

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
    #  Mysz — edytor
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
        elif self.current_tool==TOOL_SELECT and self.sel_rect_s:
            self.sel_rect_e=(x,y)
            # Bieżące zaznaczenie CP (live preview)
            self._update_cp_selection_from_rect(x,y)
        self.drawing_area.queue_draw()
        ox,oy,cw,ch=self._crect()
        t=(x-ox)/max(cw,1); a=(oy+ch/2-y)/max(ch/2,1)
        if len(self.waveform)>0:
            vlen=max(1,self.view_end-self.view_start)
            idx=int(self.view_start+t*vlen); ms=max(idx,0)/self.sample_rate*1000
            nsel=len(self.sel_cp_set)
            self._st(f"t={ms:.1f}ms  amp={a:.3f}  CP={len(self.wave_cps)} "
                     f"(zaznaczone:{nsel})  warstwy={len(self.layers)}")

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
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            # Ctrl+klik: toggle warstwy
            l=self._layer_at(x,y)
            if l: l.selected=not l.selected
        else:
            # Czyść zaznaczenie warstw i CP
            for l in self.layers: l.selected=False
            self.sel_cp_set.clear()
            self.sel_rect_s=(x,y); self.sel_rect_e=(x,y)
        self.drawing_area.queue_draw()

    def _update_cp_selection_from_rect(self, x2, y2):
        """Zaznacza CP wewnątrz prostokąta sel_rect_s / (x2,y2)."""
        if not self.sel_rect_s:
            return
        x1, y1 = self.sel_rect_s
        rx0, rx1 = min(x1,x2), max(x1,x2)
        ry0, ry1 = min(y1,y2), max(y1,y2)
        ox, oy, cw, ch = self._crect()
        vlen = max(1, self.view_end - self.view_start)
        self.sel_cp_set.clear()
        for i, cp in enumerate(self.wave_cps):
            idx, amp = cp[0], cp[1]
            t_view = (idx - self.view_start) / vlen
            px = ox + t_view * cw
            py = oy + ch/2 - amp * ch/2
            if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                self.sel_cp_set.add(i)

    def _sel_release(self, x, y):
        if self.sel_rect_s:
            self._update_cp_selection_from_rect(x, y)
            # Zaznacz też warstwy korekcji wewnątrz prostokąta
            x1,y1=self.sel_rect_s
            rx0,rx1=min(x1,x),max(x1,x); ry0,ry1=min(y1,y),max(y1,y)
            for l in self.layers:
                for px,py in [l.p0, l.p1]:
                    if rx0<=px<=rx1 and ry0<=py<=ry1:
                        l.selected=True
            self.sel_rect_s=None; self.sel_rect_e=None
        nsel=len(self.sel_cp_set)
        if nsel > 0:
            self._st(f"Zaznaczono {nsel} punktów CP")

    def _layer_at(self, x, y, thresh=14):
        for l in reversed(self.layers):
            mx=(l.p0[0]+l.p1[0])/2; my=(l.p0[1]+l.p1[1])/2
            if abs(mx-x)<thresh*3 and abs(my-y)<thresh*3: return l
        return None

    def _transform_sel(self, action):
        """Transformacja zaznaczonych warstw korekcji."""
        sel=[l for l in self.layers if l.selected]
        if not sel: self._st("Brak zaznaczonych warstw!"); return
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
            for lbl,lt in [("~ Bézier",LINE_BEZIER),("∕ Liniowy",LINE_LINEAR),("⌐ Schodek",LINE_STEP)]:
                item=Gtk.MenuItem(label=lbl)
                item.connect("activate",lambda w,s=l,t=lt:[setattr(s,'line_type',t),
                             self.drawing_area.queue_draw(),self._rerender()])
                menu.append(item)
            for lbl,cm in [("× MUL",CORR_MUL),("± ADD",CORR_ADD)]:
                item=Gtk.MenuItem(label=lbl)
                item.connect("activate",lambda w,s=l,c=cm:[setattr(s,'mode',c),
                             self.drawing_area.queue_draw(),self._rerender()])
                menu.append(item)
            menu.append(Gtk.SeparatorMenuItem())
            item=Gtk.MenuItem(label="✕ Usuń warstwę")
            item.connect("activate",lambda w,s=l:[self.layers.remove(s) if s in self.layers else None,
                         self.drawing_area.queue_draw(),self._rerender()])
            menu.append(item)
        else:
            item=Gtk.MenuItem(label="🗑 Wyczyść warstwy")
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
    def _vv2y(self,v):
        ox,oy,cw,ch=self._vrect()
        # v w zakresie 0.25..4.0, log-skala, 1.0 = środek
        lo,hi=math.log(0.25),math.log(4.0)
        return oy+ch*(1-(math.log(max(v,0.01))-lo)/(hi-lo))
    def _vx2t(self,x): ox,oy,cw,ch=self._vrect(); return max(0.,min(1.,(x-ox)/max(cw,1)))
    def _vy2v(self,y):
        ox,oy,cw,ch=self._vrect()
        lo,hi=math.log(0.25),math.log(4.0)
        lv=lo+(hi-lo)*(1-(y-oy)/max(ch,1))
        return max(0.25,min(4.0,math.exp(lv)))

    # ══════════════════════════════════════════════════════════════
    #  Kolory stref
    # ══════════════════════════════════════════════════════════════
    ZONE_COLORS = [
        (0.90, 0.35, 0.15), (0.20, 0.70, 0.90), (0.30, 0.85, 0.40),
        (0.85, 0.20, 0.70), (0.90, 0.85, 0.15), (0.55, 0.30, 0.95),
        (0.15, 0.80, 0.70), (0.95, 0.55, 0.20),
    ]

    PASS_MODES = [
        ("fwd",            "→"),
        ("rev",            "←"),
        ("fwd_half_start", "→½s"),
        ("fwd_half_end",   "→½e"),
        ("rev_half_start", "←½s"),
        ("rev_half_end",   "←½e"),
        ("pingpong",       "↔"),
        ("custom",         "✎"),
    ]

    def _zone_color(self, idx):
        return self.ZONE_COLORS[idx % len(self.ZONE_COLORS)]

    def _zrect(self):
        w = self.zone_area.get_allocated_width()
        h = self.zone_area.get_allocated_height()
        return self.VEL_PL, 4, w - self.VEL_PL - self.VEL_PR, h - 8

    def _zt2x(self, t):
        ox, oy, cw, ch = self._zrect(); return ox + t * cw

    def _zx2t(self, x):
        ox, oy, cw, ch = self._zrect()
        return max(0.0, min(1.0, (x - ox) / max(cw, 1)))

    # ── rysowanie stref ───────────────────────────────────────────
    def _zone_draw(self, widget, cr):
        ox, oy, cw, ch = self._zrect()

        cr.set_source_rgb(0.10, 0.10, 0.12); cr.paint()
        cr.set_source_rgb(0.14, 0.14, 0.17)
        cr.rectangle(ox, oy, cw, ch); cr.fill()

        # miniatura waveformu
        if len(self.waveform) > 1:
            cr.set_source_rgba(0.25, 0.55, 0.28, 0.5); cr.set_line_width(0.8)
            step = len(self.waveform) / cw; first = True
            for i in range(int(cw)):
                idx = min(int(i * step), len(self.waveform)-1)
                y = oy + ch/2 - self.waveform[idx] * (ch/2 - 4)
                if first: cr.move_to(ox + i, y); first = False
                else:     cr.line_to(ox + i, y)
            cr.stroke()

        cr.set_source_rgba(0.4, 0.4, 0.45, 0.4); cr.set_line_width(0.5)
        cr.move_to(ox, oy+ch/2); cr.line_to(ox+cw, oy+ch/2); cr.stroke()

        # Rysuj strefy z typem przenikania
        lane_h = max(6, (ch - 20) // max(len(self.zones), 1))
        lane_h = min(lane_h, 22)

        for zi, zone in enumerate(self.zones):
            r, g, b = self._zone_color(zi)
            x0 = self._zt2x(zone['t0'])
            x1 = self._zt2x(zone['t1'])
            zy  = oy + 4 + zi * (lane_h + 2)
            sel = (zi == self.zone_sel_idx)
            xfade = zone.get('crossfade', self.zone_crossfade)

            if xfade == 'crossfade' and zi > 0:
                # Zachodzące — narysuj gradient nakładania
                prev_zone = self.zones[zi-1]
                px1 = self._zt2x(prev_zone['t1'])
                if px1 > x0:  # strefy zachodzą
                    # gradient w obszarze nałożenia
                    ovx0 = x0; ovx1 = px1
                    cr.set_source_rgba(r, g, b, 0.20)
                    cr.rectangle(ovx0, zy-2, ovx1-ovx0, lane_h+4); cr.fill()
                    # linia przenikania
                    cr.set_source_rgba(1,1,1,0.4); cr.set_line_width(1); cr.set_dash([2,2])
                    mid_x = (ovx0+ovx1)/2
                    cr.move_to(mid_x, zy); cr.line_to(mid_x, zy+lane_h); cr.stroke()
                    cr.set_dash([])

            # blok strefy
            cr.set_source_rgba(r, g, b, 0.35 if not sel else 0.55)
            cr.rectangle(x0, zy, x1-x0, lane_h); cr.fill()
            cr.set_source_rgba(r, g, b, 0.9 if sel else 0.6)
            cr.set_line_width(2 if sel else 1)
            cr.rectangle(x0, zy, x1-x0, lane_h); cr.stroke()

            # Ikona przenikania przy lewej krawędzi
            if xfade == 'crossfade':
                cr.set_source_rgba(1,0.8,0,0.9); cr.set_font_size(7)
                cr.move_to(x0+1, zy+8); cr.show_text("⋊")
            else:
                cr.set_source_rgba(0.6,0.6,0.7,0.7); cr.set_font_size(7)
                cr.move_to(x0+1, zy+8); cr.show_text("|")

            # uchwyty granic
            for hx in [x0, x1]:
                cr.set_source_rgba(1, 1, 1, 0.8)
                cr.rectangle(hx - 3, zy, 6, lane_h); cr.fill()

            # mini ikony przebiegów
            passes = zone.get('passes', [{'mode':'fwd','s':0.0,'e':1.0}])
            rep    = zone.get('repeats', 1)
            pw     = max(4, (x1 - x0 - 4) / max(len(passes) * rep, 1))
            cr.set_font_size(7)
            for ri in range(rep):
                for pi, pas in enumerate(passes):
                    px = x0 + 2 + (ri * len(passes) + pi) * pw
                    if px + pw > x1 - 1: break
                    alpha = 0.9 if ri % 2 == 0 else 0.6
                    cr.set_source_rgba(r, g, b, alpha)
                    cr.rectangle(px, zy+2, pw-1, lane_h-4); cr.fill()
                    lbl = self._pass_icon(pas['mode'])
                    cr.set_source_rgba(0, 0, 0, 0.9)
                    cr.move_to(px+1, zy + lane_h - 3); cr.show_text(lbl)

            # Etykieta
            label = f"Z{zi+1}×{rep}"
            cr.set_font_size(8); cr.set_source_rgba(1, 1, 1, 0.9)
            cr.move_to(x0 + 4, zy + lane_h - 3); cr.show_text(label)

        # End-to marker w obszarze stref
        if len(self.waveform) > 1:
            ex = self._zt2x(self.play_end_frac)
            cr.set_source_rgba(1.0, 0.3, 0.3, 0.9)
            cr.set_line_width(2); cr.set_dash([4,2])
            cr.move_to(ex, oy); cr.line_to(ex, oy+ch); cr.stroke()
            cr.set_dash([])
            cr.set_font_size(7); cr.set_source_rgba(1,0.3,0.3,1)
            cr.move_to(ex+2, oy+10); cr.show_text("END")

        # Oś czasu
        cr.set_font_size(8); cr.set_source_rgb(0.6, 0.6, 0.65)
        for i in range(9):
            x = self._zt2x(i / 8)
            cr.move_to(x, oy+ch-8); cr.line_to(x, oy+ch)
            cr.set_line_width(0.5); cr.stroke()
            cr.move_to(x+1, oy+ch-1)
            if len(self.waveform) > 0:
                ms = (i/8) * len(self.waveform) / self.sample_rate * 1000
                cr.show_text(f"{ms:.0f}ms")

        if self.zone_sel_idx is not None:
            self._zone_draw_pass_panel(cr, ox, oy, cw, ch)

    def _pass_icon(self, mode):
        icons = {'fwd':'→','rev':'←','fwd_half_start':'→½s','fwd_half_end':'→½e',
                 'rev_half_start':'←½s','rev_half_end':'←½e','pingpong':'↔','custom':'✎'}
        return icons.get(mode, '?')

    def _zone_draw_pass_panel(self, cr, ox, oy, cw, ch):
        zi = self.zone_sel_idx
        if zi is None or zi >= len(self.zones): return
        zone = self.zones[zi]; passes = zone.get('passes', [])
        r, g, b = self._zone_color(zi)
        panel_y = oy + ch - 18
        cr.set_source_rgba(0.08, 0.08, 0.12, 0.92)
        cr.rectangle(ox, panel_y, cw, 18); cr.fill()
        cr.set_font_size(8); cr.set_source_rgba(r, g, b, 1)
        cr.move_to(ox+2, panel_y+12)
        cr.show_text(f"Z{zi+1} przebiegi ({len(passes)}):")
        for pi, pas in enumerate(passes):
            px = ox + 80 + pi * 52
            cr.set_source_rgba(r, g, b, 0.7)
            cr.rectangle(px, panel_y+1, 50, 16); cr.fill()
            cr.set_source_rgba(1,1,1,0.9)
            icon = self._pass_icon(pas['mode'])
            s_pct = int(pas.get('s',0)*100); e_pct = int(pas.get('e',1)*100)
            cr.move_to(px+2, panel_y+12)
            cr.show_text(f"P{pi+1}:{icon} {s_pct}-{e_pct}%")

    # ── mysz stref ────────────────────────────────────────────────
    HANDLE_W = 6

    def _zone_hit(self, x, oy, ch, zi, zone):
        x0 = self._zt2x(zone['t0']); x1 = self._zt2x(zone['t1'])
        if abs(x - x0) <= self.HANDLE_W: return 'left'
        if abs(x - x1) <= self.HANDLE_W: return 'right'
        if x0 <= x <= x1:               return 'body'
        return None

    def _zone_press(self, widget, event):
        x, y = event.x, event.y
        ox, oy, cw, ch = self._zrect()
        if event.button == 3:
            for zi, zone in enumerate(self.zones):
                if self._zone_hit(x, oy, ch, zi, zone):
                    self.zones.pop(zi)
                    if self.zone_sel_idx == zi: self.zone_sel_idx = None
                    elif self.zone_sel_idx and self.zone_sel_idx > zi:
                        self.zone_sel_idx -= 1
                    self.zone_area.queue_draw()
                    self._zone_refresh_pass_ui()
                    return
        if event.button == 1:
            for zi, zone in enumerate(self.zones):
                hit = self._zone_hit(x, oy, ch, zi, zone)
                if hit:
                    self.zone_sel_idx = zi
                    self.zone_drag = (zi, hit, x, zone['t0'], zone['t1'])
                    self.zone_area.queue_draw()
                    self._zone_refresh_pass_ui()
                    return
            t = self._zx2t(x)
            new_zone = {
                't0': t, 't1': min(t+0.1, 1.0),
                'repeats': 1,
                'passes': [{'mode':'fwd','s':0.0,'e':1.0}],
                'crossfade': self.zone_crossfade,
            }
            self.zones.append(new_zone)
            self.zone_sel_idx = len(self.zones)-1
            self.zone_drag = (self.zone_sel_idx, 'right', x, t, min(t+0.1,1.0))
            self.zone_area.queue_draw()
            self._zone_refresh_pass_ui()

    def _zone_motion(self, widget, event):
        if self.zone_drag is None: return
        zi, hit, x_start, t0_start, t1_start = self.zone_drag
        dx_t = self._zx2t(event.x) - self._zx2t(x_start)
        zone = self.zones[zi]; MIN_W = 0.01
        if hit == 'left':
            zone['t0'] = max(0.0, min(t0_start + dx_t, zone['t1'] - MIN_W))
        elif hit == 'right':
            zone['t1'] = max(zone['t0'] + MIN_W, min(1.0, t1_start + dx_t))
        elif hit == 'body':
            span = t1_start - t0_start
            nt0 = max(0.0, min(1.0 - span, t0_start + dx_t))
            zone['t0'] = nt0; zone['t1'] = nt0 + span
        self.zone_area.queue_draw()

    def _zone_release(self, widget, event):
        self.zone_drag = None
        self.zone_area.queue_draw()
        self.store.update_zones(self.zones)

    # ── panel przebiegów ──────────────────────────────────────────
    def _zone_refresh_pass_ui(self):
        for child in self.pass_box.get_children():
            self.pass_box.remove(child)
        zi = self.zone_sel_idx
        if zi is None or zi >= len(self.zones):
            self.pass_box.show_all(); return
        zone = self.zones[zi]; passes = zone.get('passes', [])
        r, g, b = self._zone_color(zi)

        lbl = Gtk.Label()
        lbl.set_markup(f'<span foreground="#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"><b>Z{zi+1}:</b></span>')
        self.pass_box.pack_start(lbl, False, False, 4)

        # Typ przenikania dla tej konkretnej strefy
        xf_lbl = Gtk.Label(label="⋊:")
        xf_lbl.set_tooltip_text("Przenikanie tej strefy")
        self.pass_box.pack_start(xf_lbl, False, False, 0)
        xf_combo = Gtk.ComboBoxText()
        xf_combo.append_text("ostre"); xf_combo.append_text("zachodzące")
        cur_xf = zone.get('crossfade', self.zone_crossfade)
        xf_combo.set_active(0 if cur_xf == 'sharp' else 1)
        xf_combo.connect("changed", self._make_xfade_cb(zi))
        self.pass_box.pack_start(xf_combo, False, False, 0)
        self.pass_box.pack_start(_sep(), False,False,4)

        for pi, pas in enumerate(passes):
            pf = Gtk.Frame(); pf.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
            phb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
            phb.set_border_width(2); pf.add(phb)
            phb.pack_start(Gtk.Label(label=f"P{pi+1}:"), False, False, 0)
            mc = Gtk.ComboBoxText()
            for mode_id, mode_lbl in self.PASS_MODES:
                mc.append_text(mode_lbl)
            mode_ids = [m[0] for m in self.PASS_MODES]
            cur_mode = pas.get('mode','fwd')
            if cur_mode in mode_ids: mc.set_active(mode_ids.index(cur_mode))
            mc.connect("changed", self._make_mode_cb(zi, pi, mode_ids))
            phb.pack_start(mc, False, False, 0)
            phb.pack_start(Gtk.Label(label="od:"), False, False, 0)
            s_spin = Gtk.SpinButton()
            s_spin.set_range(0,99); s_spin.set_value(int(pas.get('s',0)*100))
            s_spin.set_increments(5,10); s_spin.set_width_chars(3)
            s_spin.connect("value-changed", self._make_range_cb(zi, pi, 's'))
            phb.pack_start(s_spin, False, False, 0)
            phb.pack_start(Gtk.Label(label="% do:"), False, False, 0)
            e_spin = Gtk.SpinButton()
            e_spin.set_range(1,100); e_spin.set_value(int(pas.get('e',1)*100))
            e_spin.set_increments(5,10); e_spin.set_width_chars(3)
            e_spin.connect("value-changed", self._make_range_cb(zi, pi, 'e'))
            phb.pack_start(e_spin, False, False, 0)
            phb.pack_start(Gtk.Label(label="%"), False, False, 0)
            self.pass_box.pack_start(pf, False, False, 0)

        self.pass_box.show_all()

    def _make_xfade_cb(self, zi):
        def cb(combo):
            mapping = {"ostre":"sharp","zachodzące":"crossfade"}
            val = mapping.get(combo.get_active_text(),"sharp")
            if zi < len(self.zones):
                self.zones[zi]['crossfade'] = val
                self.zone_area.queue_draw()
        return cb

    def _make_mode_cb(self, zi, pi, mode_ids):
        def cb(combo):
            idx = combo.get_active()
            if 0 <= idx < len(mode_ids) and zi < len(self.zones):
                self.zones[zi]['passes'][pi]['mode'] = mode_ids[idx]
                self.zone_area.queue_draw()
        return cb

    def _make_range_cb(self, zi, pi, key):
        def cb(spin):
            if zi < len(self.zones) and pi < len(self.zones[zi]['passes']):
                self.zones[zi]['passes'][pi][key] = spin.get_value() / 100.0
                self.zone_area.queue_draw()
        return cb

    def _zone_repeats_changed(self, spin):
        zi = self.zone_sel_idx
        if zi is not None and zi < len(self.zones):
            self.zones[zi]['repeats'] = int(spin.get_value())
            self.zone_area.queue_draw()

    def _zone_add_pass(self, widget=None):
        zi = self.zone_sel_idx
        if zi is None or zi >= len(self.zones): return
        self.zones[zi]['passes'].append({'mode':'fwd','s':0.0,'e':1.0})
        self._zone_refresh_pass_ui(); self.zone_area.queue_draw()

    def _zone_del_pass(self, widget=None):
        zi = self.zone_sel_idx
        if zi is None or zi >= len(self.zones): return
        passes = self.zones[zi]['passes']
        if len(passes) > 1: passes.pop()
        self._zone_refresh_pass_ui(); self.zone_area.queue_draw()

    def _zone_clear(self, widget=None):
        self.zones.clear(); self.zone_sel_idx=None
        self._zone_refresh_pass_ui(); self.zone_area.queue_draw()

    # ── renderowanie stref → fala z crossfade ─────────────────────
    def _apply_crossfade(self, parts: list, fade_samples: int = 512) -> np.ndarray:
        """Łączy fragmenty z fade-in/out na złączach."""
        if not parts:
            return np.array([])
        if len(parts) == 1:
            return parts[0]
        result = parts[0].copy()
        for chunk in parts[1:]:
            fl = min(fade_samples, len(result), len(chunk))
            if fl > 1:
                fade_out = np.linspace(1.0, 0.0, fl)
                fade_in  = np.linspace(0.0, 1.0, fl)
                result[-fl:] = result[-fl:] * fade_out + chunk[:fl] * fade_in
                result = np.concatenate([result, chunk[fl:]])
            else:
                result = np.concatenate([result, chunk])
        return result

    def _zone_extract_pass(self, wave, zone, pas):
        n = len(wave)
        i0 = int(zone['t0'] * n); i1 = int(zone['t1'] * n)
        if i1 <= i0: return np.array([])
        seg = wave[i0:i1]; seg_n = len(seg)
        s_frac = pas.get('s', 0.0); e_frac = pas.get('e', 1.0)
        si = int(s_frac * seg_n); ei = int(e_frac * seg_n)
        si = max(0, min(si, seg_n-1)); ei = max(si+1, min(ei, seg_n))
        mode = pas.get('mode','fwd')
        if   mode == 'fwd':            chunk = seg[si:ei]
        elif mode == 'rev':            chunk = seg[si:ei][::-1]
        elif mode == 'fwd_half_start': chunk = seg[si:si+(ei-si)//2]
        elif mode == 'fwd_half_end':   chunk = seg[si+(ei-si)//2:ei]
        elif mode == 'rev_half_start': chunk = seg[si:si+(ei-si)//2][::-1]
        elif mode == 'rev_half_end':   chunk = seg[si+(ei-si)//2:ei][::-1]
        elif mode == 'pingpong':       chunk = np.concatenate([seg[si:ei], seg[si:ei][::-1]])
        elif mode == 'custom':         chunk = seg[si:ei]
        else:                          chunk = seg[si:ei]
        return chunk

    def _zone_build_wave(self):
        wave = self._wave_trimmed()
        if len(wave) < 2: return np.array([])
        mode = self.zone_mode
        use_xfade = (self.zone_crossfade == 'crossfade')

        if mode == 'sequential':
            parts = []
            for zone in self.zones:
                rep    = zone.get('repeats', 1)
                passes = zone.get('passes', [{'mode':'fwd','s':0.0,'e':1.0}])
                zone_xf = zone.get('crossfade', self.zone_crossfade) == 'crossfade'
                zone_parts = []
                for _ in range(rep):
                    for pas in passes:
                        chunk = self._zone_extract_pass(wave, zone, pas)
                        if len(chunk): zone_parts.append(chunk)
                if zone_parts:
                    # przenikanie wewnątrz strefy
                    if zone_xf and len(zone_parts) > 1:
                        parts.append(self._apply_crossfade(zone_parts, 256))
                    else:
                        parts.append(np.concatenate(zone_parts))
            if not parts: return wave.copy()
            # przenikanie między strefami
            if use_xfade and len(parts) > 1:
                return self._apply_crossfade(parts, 512)
            return np.concatenate(parts)

        elif mode == 'layer':
            n = len(wave); out = np.zeros(n); weight = np.zeros(n)
            for zone in self.zones:
                i0 = int(zone['t0']*n); i1 = int(zone['t1']*n)
                if i1 <= i0: continue
                rep    = zone.get('repeats', 1)
                passes = zone.get('passes', [{'mode':'fwd','s':0.0,'e':1.0}])
                zone_chunks = []
                for _ in range(rep):
                    for pas in passes:
                        c = self._zone_extract_pass(wave, zone, pas)
                        if len(c): zone_chunks.append(c)
                if not zone_chunks: continue
                blended = np.concatenate(zone_chunks)
                seg_len = len(out[i0:i1])  # actual slot length (may be < i1-i0 near end)
                if len(blended) >= seg_len: blended = blended[:seg_len]
                else: blended = np.resize(blended, seg_len)
                # Crossfade przy nakładaniu
                xf = zone.get('crossfade', self.zone_crossfade) == 'crossfade'
                if xf:
                    env = np.ones(seg_len)
                    fl = min(256, seg_len//4)
                    if fl > 1:
                        env[:fl]  = np.linspace(0, 1, fl)
                        env[-fl:] = np.linspace(1, 0, fl)
                    blended *= env
                out[i0:i1] += blended; weight[i0:i1] += 1
            no_zone = weight == 0
            out[no_zone] = wave[no_zone]; weight[no_zone] = 1
            out /= weight
            mx = np.max(np.abs(out))
            return out/mx if mx > 1e-9 else out

        elif mode == 'nested':
            n = len(wave); out = wave.copy()
            for zone in sorted(self.zones, key=lambda z: abs(z['t1']-z['t0']), reverse=True):
                i0 = int(zone['t0']*n); i1 = int(zone['t1']*n)
                if i1 <= i0: continue
                rep    = zone.get('repeats', 1)
                passes = zone.get('passes', [{'mode':'fwd','s':0.0,'e':1.0}])
                chunks = []
                for _ in range(rep):
                    for pas in passes:
                        c = self._zone_extract_pass(wave, zone, pas)
                        if len(c): chunks.append(c)
                if not chunks: continue
                blended = np.concatenate(chunks)
                seg_len = len(out[i0:i1])  # actual slot length
                if len(blended) >= seg_len: blended = blended[:seg_len]
                else: blended = np.resize(blended, seg_len)
                xf = zone.get('crossfade', self.zone_crossfade) == 'crossfade'
                if xf:
                    fl = min(128, seg_len//4)
                    if fl > 1:
                        env = np.ones(seg_len)
                        env[:fl]  = np.linspace(0,1,fl)
                        env[-fl:] = np.linspace(1,0,fl)
                        blended *= env
                out[i0:i1] = blended
            mx = np.max(np.abs(out))
            return out/mx if mx > 1e-9 else out

        return wave.copy()

    def _zone_play(self, widget=None):
        if not PYGAME_OK or len(self.waveform) < 2: return
        result = self._zone_build_wave()
        if len(result) < 2: return
        self.zone_rendered = result
        self._play_wave(result)
        self._st(f"Strefy odtworzone: {len(result)} próbek  "
                 f"{len(result)/self.sample_rate*1000:.0f}ms  "
                 f"[{self.zone_crossfade}]")

    def _zone_stop(self, widget=None):
        if PYGAME_OK: pygame.mixer.music.stop()

    def _zone_render_to_wave(self, widget=None):
        result = self._zone_build_wave()
        if len(result) < 2: return
        self._set_wave(result, source="zone_render")
        self._st(f"Strefy → nowa fala: {len(result)} próbek")

    # ══════════════════════════════════════════════════════════════
    #  Velocity draw
    # ══════════════════════════════════════════════════════════════
    def _vel_draw(self, widget, cr):
        ox,oy,cw,ch=self._vrect()
        cr.set_source_rgb(0.10,0.10,0.12); cr.paint()
        cr.set_source_rgb(0.14,0.14,0.17); cr.rectangle(ox,oy,cw,ch); cr.fill()
        cr.set_font_size(8)
        for spd,lbl in [(4.0,"4×"),(2.0,"2×"),(1.0,"1×"),(0.5,"½×"),(0.25,"¼×")]:
            y=self._vv2y(spd)
            alpha=0.7 if spd==1.0 else 0.35
            cr.set_source_rgba(0.3,0.3,0.35,alpha); cr.set_line_width(0.5 if spd!=1.0 else 1.2)
            cr.move_to(ox,y); cr.line_to(ox+cw,y); cr.stroke()
            cr.set_source_rgb(0.55,0.55,0.55); cr.move_to(2,y+3); cr.show_text(lbl)
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
            cr.move_to(ox+cw/2-40,oy+ch/2+5); cr.show_text("SPEED OFF")

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
        self.store.update_vel(self.vel_nodes)

    def _vel_release(self, widget, event):
        self.vel_drag_idx=None; self.vel_area.queue_draw()

    def _vel_reset(self, w=None):
        self.vel_nodes=[[0.0,1.0],[1.0,1.0]]
        self.vel_area.queue_draw(); self._rerender()

    def _vel_curve(self, n):
        """Zwraca krzywą prędkości (speed) w n punktach. Wartości: 0.25..4.0, 1.0=normalna."""
        nodes=sorted(self.vel_nodes,key=lambda nd:nd[0])
        ts=np.array([nd[0] for nd in nodes]); vs=np.array([nd[1] for nd in nodes])
        return np.interp(np.linspace(0.,1.,n),ts,vs)

    def _apply_speed_warp(self, wave):
        """Przepróbkowuje falę zgodnie z krzywą prędkości.
        v>1 = przyspieszenie (wyjście krótsze), v<1 = zwolnienie (wyjście dłuższe).
        Długość wyjściowa = n / mean(speed), zaokrąglona do int."""
        n = len(wave)
        if n < 2: return wave
        speed = self._vel_curve(n)           # prędkość w każdym punkcie wejścia
        # Cumsum pozycji wyjściowej: całka prędkości po czasie
        # position[i] = ile próbek wyjściowych odpowiada wejściu [0..i]
        pos = np.cumsum(speed)               # rosnąca, pos[-1] = suma prędkości
        pos = pos / pos[-1]                  # normalizuj do [0..1]
        n_out = max(2, int(round(n / np.mean(speed))))
        t_out = np.linspace(0., 1., n_out)
        # Odwróć mapowanie: dla każdego t_out znajdź odpowiedni indeks wejścia
        t_in = np.interp(t_out, pos, np.linspace(0., 1., n))
        return np.interp(t_in * (n-1), np.arange(n), wave)

    # ══════════════════════════════════════════════════════════════
    #  Odtwarzanie
    # ══════════════════════════════════════════════════════════════
    def on_play(self, w=None):
        if len(self.waveform)<2 or not PYGAME_OK: return
        self._play_wave(self._wave_trimmed())

    def on_stop(self, w=None):
        if PYGAME_OK: pygame.mixer.music.stop()

    def _toggle_live(self, w=None):
        self.live_play=not self.live_play
        self._st("Live: "+("ON" if self.live_play else "OFF"))

    def _play_now(self):
        if len(self.waveform)<2 or not PYGAME_OK: return
        self._play_wave(self._wave_trimmed())

    def _play_wave(self, wave: np.ndarray):
        if not PYGAME_OK or len(wave) < 2: return
        p = self._tmp(wave)
        pygame.mixer.music.load(p)
        pygame.mixer.music.play()

    def _tmp(self, wave):
        p=os.path.join(tempfile.gettempdir(),"ed_waves4.wav")
        wav.write(p,self.sample_rate,(wave*32767).astype(np.int16)); return p

    # ══════════════════════════════════════════════════════════════
    #  Import / Eksport
    # ══════════════════════════════════════════════════════════════
    def on_save_wav(self, widget):
        dlg=Gtk.FileChooserDialog(title="Zapisz WAV",parent=self,action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,Gtk.STOCK_SAVE,Gtk.ResponseType.OK)
        if dlg.run()==Gtk.ResponseType.OK:
            wav.write(dlg.get_filename(),self.sample_rate,(self._wave_trimmed()*32767).astype(np.int16))
        dlg.destroy()

    def on_import_wav(self, widget):
        dlg=Gtk.FileChooserDialog(title="Wczytaj WAV",parent=self,action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,Gtk.STOCK_OPEN,Gtk.ResponseType.OK)
        if dlg.run()==Gtk.ResponseType.OK:
            sr,data=wav.read(dlg.get_filename())
            if len(data.shape)==2: data=data.mean(axis=1)
            data=data.astype(float); mx=np.max(np.abs(data))
            if mx>0: data/=mx
            self.sample_rate=sr
            self._set_wave(data, source="import", wtype="wav")
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
