import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo
import numpy as np
import pygame
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
import random
import tempfile
import os
import threading

# ──────────────────────────────────────────────────────────────────
#  Stałe
# ──────────────────────────────────────────────────────────────────
MIN_FREQ    = 5.0
MAX_FREQ    = 25000.0
RULER_H     = 28          # wysokość linijki czasu (px)
FREQ_W      = 58          # szerokość panelu osi częstotliwości (px)
CP_RADIUS   = 6           # promień węzła kontrolnego
SEL_THRESH  = 14          # próg kliknięcia w węzeł (px)

LINE_BEZIER  = "bezier"
LINE_LINEAR  = "linear"
LINE_STEP    = "step"

TOOL_DRAW    = "draw"
TOOL_EDIT    = "edit"
TOOL_SELECT  = "select"


# ──────────────────────────────────────────────────────────────────
#  Klasa segmentu krzywej
# ──────────────────────────────────────────────────────────────────
class Segment:
    """Para węzłów p0→p1 + typ linii + uchwyty Béziera cp0/cp1."""
    def __init__(self, p0, p1, line_type=LINE_BEZIER):
        self.p0 = list(p0)
        self.p1 = list(p1)
        self.line_type = line_type
        dx = (p1[0] - p0[0]) / 3.0
        self.cp0 = [p0[0] + dx, p0[1]]
        self.cp1 = [p1[0] - dx, p1[1]]
        self.selected = False


# ──────────────────────────────────────────────────────────────────
#  Główna klasa
# ──────────────────────────────────────────────────────────────────
class WaveformEditor(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Waveform Editor v2")
        self.set_default_size(1360, 720)
        self.connect("destroy", Gtk.main_quit)

        # ── stan ──────────────────────────────────────────────────
        self.segments           = []
        self.waveform           = np.array([])
        self.original_waveform  = np.array([])
        self.sample_rate        = 44100
        self.zoom_level         = 1.0
        self.view_start         = 0
        self.view_end           = 0

        self.current_tool       = TOOL_DRAW
        self.current_line       = LINE_BEZIER

        self.draw_start         = None      # (x,y) początku rysowanego segmentu
        self.mouse_pos          = (0, 0)    # bieżąca pozycja myszy

        self.drag_target        = None      # (idx_seg, attr_str)
        self.drag_offset        = (0, 0)

        self.sel_rect_start     = None
        self.sel_rect_end       = None

        self.sample_width_px    = 2
        self.show_freq_axis     = True
        self.freq_view_min      = MIN_FREQ
        self.freq_view_max      = MAX_FREQ

        self.live_play_active   = False

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)

        self._build_ui()

    # ══════════════════════════════════════════════════════════════
    #  Budowa UI
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add(vbox)

        # ── pasek 1: narzędzia + typy linii + akcje ──────────────
        tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        tb.set_border_width(4)
        vbox.pack_start(tb, False, False, 0)

        # narzędzia – RadioButton
        tool_group = None
        for lbl, tool in [("✏ Draw", TOOL_DRAW), ("✋ Edit", TOOL_EDIT), ("⬚ Select", TOOL_SELECT)]:
            btn = Gtk.RadioButton.new_with_label_from_widget(tool_group, lbl)
            if tool_group is None:
                tool_group = btn
            btn.connect("toggled", self._on_tool_changed, tool)
            tb.pack_start(btn, False, False, 0)

        tb.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        # typy linii
        line_group = None
        for lbl, lt in [("~ Bézier", LINE_BEZIER), ("/ Linia", LINE_LINEAR), ("⌐ Schodek", LINE_STEP)]:
            btn = Gtk.RadioButton.new_with_label_from_widget(line_group, lbl)
            if line_group is None:
                line_group = btn
            btn.connect("toggled", self._on_line_type_changed, lt)
            tb.pack_start(btn, False, False, 0)

        tb.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        # przyciski akcji
        for lbl, cb in [
            ("▶ Play",       self.on_play),
            ("⏹ Stop",       self.on_stop),
            ("⟳ Live",       self._toggle_live_play),
            ("💾 Zapis WAV", self.on_save_wav),
            ("📂 Import WAV",self.on_import_wav),
            ("🎲 Losowa",    self.on_generate_random_wave),
            ("🗑 Wyczyść",   self._clear_all),
        ]:
            b = Gtk.Button(label=lbl)
            b.connect("clicked", cb)
            tb.pack_start(b, False, False, 0)

        tb.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)
        tb.pack_start(Gtk.Label(label="Szer.próbki[px]:"), False, False, 0)
        self.sw_spin = Gtk.SpinButton()
        self.sw_spin.set_range(1, 20)
        self.sw_spin.set_value(self.sample_width_px)
        self.sw_spin.set_increments(1, 2)
        self.sw_spin.connect("value-changed",
                             lambda s: setattr(self, 'sample_width_px', int(s.get_value())))
        tb.pack_start(self.sw_spin, False, False, 0)

        # ── pasek 2: transformacje zaznaczenia ───────────────────
        tb2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        tb2.set_border_width(2)
        vbox.pack_start(tb2, False, False, 0)

        tb2.pack_start(Gtk.Label(label="Zaznaczenie:"), False, False, 0)

        for lbl, act in [
            ("Ścisn.Y",    'compress_y'),
            ("Rozciąg.Y",  'stretch_y'),
            ("Ścisn.X",    'compress_x'),
            ("Rozciąg.X",  'stretch_x'),
            ("Odwróć Y",   'flip_y'),
            ("Odwróć X",   'flip_x'),
            ("Usuń",       'delete'),
        ]:
            b = Gtk.Button(label=lbl)
            b.connect("clicked", lambda w, a=act: self._transform_selection(a))
            tb2.pack_start(b, False, False, 0)

        tb2.pack_start(Gtk.Label(label="  Skala Y:"), False, False, 0)
        self.sel_scale_y = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_scale_y.set_range(0.05, 4.0)
        self.sel_scale_y.set_value(1.0)
        self.sel_scale_y.set_digits(2)
        self.sel_scale_y.set_size_request(110, -1)
        tb2.pack_start(self.sel_scale_y, False, False, 0)

        tb2.pack_start(Gtk.Label(label="  Skala X:"), False, False, 0)
        self.sel_scale_x = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sel_scale_x.set_range(0.05, 4.0)
        self.sel_scale_x.set_value(1.0)
        self.sel_scale_x.set_digits(2)
        self.sel_scale_x.set_size_request(110, -1)
        tb2.pack_start(self.sel_scale_x, False, False, 0)

        # ── obszar rysowania + scrollbar ──────────────────────────
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(hbox, True, True, 0)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(1200, 460)
        self.drawing_area.connect("draw", self.on_draw)
        self.drawing_area.connect("button-press-event",   self.on_button_press)
        self.drawing_area.connect("button-release-event", self.on_button_release)
        self.drawing_area.connect("motion-notify-event",  self.on_motion)
        mask = (Gdk.EventMask.BUTTON_PRESS_MASK |
                Gdk.EventMask.BUTTON_RELEASE_MASK |
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.drawing_area.set_events(mask)
        hbox.pack_start(self.drawing_area, True, True, 0)

        self.vscroll = Gtk.VScrollbar()
        self.vscroll.connect("value-changed", self.on_scroll_changed)
        hbox.pack_start(self.vscroll, False, False, 0)

        # ── zoom ──────────────────────────────────────────────────
        zbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        zbox.set_border_width(2)
        vbox.pack_start(zbox, False, False, 0)
        zbox.pack_start(Gtk.Label(label="Zoom:"), False, False, 0)
        self.zoom_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.zoom_scale.set_range(1, 20)
        self.zoom_scale.set_value(1)
        self.zoom_scale.set_digits(1)
        self.zoom_scale.set_increments(0.1, 1)
        self.zoom_scale.set_hexpand(True)
        self.zoom_scale.connect("value-changed", self.on_zoom_changed)
        zbox.pack_start(self.zoom_scale, True, True, 0)

        # ── pasek statusu ─────────────────────────────────────────
        self.statusbar = Gtk.Label(label="Gotowy  |  ✏ Draw  |  ~ Bezier")
        self.statusbar.set_xalign(0)
        self.statusbar.set_border_width(3)
        vbox.pack_start(self.statusbar, False, False, 0)

    # ══════════════════════════════════════════════════════════════
    #  Zmiana narzędzia / typu linii
    # ══════════════════════════════════════════════════════════════
    def _on_tool_changed(self, btn, tool):
        if btn.get_active():
            self.current_tool = tool
            self._status(f"Narzędzie: {tool}")

    def _on_line_type_changed(self, btn, lt):
        if btn.get_active():
            self.current_line = lt
            for seg in self.segments:
                if seg.selected:
                    seg.line_type = lt
            self.drawing_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Pomocnicze współrzędne
    # ══════════════════════════════════════════════════════════════
    def _canvas_rect(self):
        w = self.drawing_area.get_allocated_width()
        h = self.drawing_area.get_allocated_height()
        ox = FREQ_W if self.show_freq_axis else 0
        oy = RULER_H
        return ox, oy, w - ox, h - oy   # (ox, oy, cw, ch)

    def _xy_to_normwave(self, x, y):
        ox, oy, cw, ch = self._canvas_rect()
        t = (x - ox) / max(cw, 1)
        a = (oy + ch / 2 - y) / max(ch / 2, 1)
        return t, a

    # ══════════════════════════════════════════════════════════════
    #  Rysowanie (on_draw)
    # ══════════════════════════════════════════════════════════════
    def on_draw(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        ox, oy, cw, ch = self._canvas_rect()

        # tło
        cr.set_source_rgb(0.11, 0.11, 0.13)
        cr.paint()

        # oś częstotliwości
        if self.show_freq_axis:
            cr.set_source_rgb(0.17, 0.17, 0.20)
            cr.rectangle(0, oy, FREQ_W, ch)
            cr.fill()
            self._draw_freq_axis(cr, oy, ch)

        # linijka czasu
        self._draw_time_ruler(cr, ox, oy, cw)

        # siatka
        self._draw_grid(cr, ox, oy, cw, ch)

        # oś środkowa
        cr.set_source_rgba(0.45, 0.45, 0.50, 0.7)
        cr.set_line_width(1)
        cr.move_to(ox, oy + ch / 2)
        cr.line_to(ox + cw, oy + ch / 2)
        cr.stroke()

        # waveform z próbek
        if len(self.waveform) > 1:
            self._draw_waveform(cr, ox, oy, cw, ch)

        # segmenty
        for seg in self.segments:
            self._draw_segment(cr, seg)

        # podgląd rysowania
        if self.draw_start is not None:
            cr.set_source_rgba(0.3, 0.9, 1.0, 0.45)
            cr.set_line_width(1.5)
            cr.set_dash([6, 4])
            cr.move_to(self.draw_start[0], self.draw_start[1])
            cr.line_to(self.mouse_pos[0], self.mouse_pos[1])
            cr.stroke()
            cr.set_dash([])
            cr.set_source_rgba(0.3, 0.9, 1.0, 0.8)
            cr.arc(self.draw_start[0], self.draw_start[1], CP_RADIUS, 0, 6.28)
            cr.fill()

        # prostokąt zaznaczenia
        if self.sel_rect_start and self.sel_rect_end:
            x1, y1 = self.sel_rect_start
            x2, y2 = self.sel_rect_end
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            cr.set_source_rgba(0.3, 0.75, 1.0, 0.12)
            cr.rectangle(rx, ry, rw, rh)
            cr.fill()
            cr.set_source_rgba(0.3, 0.75, 1.0, 0.75)
            cr.set_line_width(1)
            cr.rectangle(rx, ry, rw, rh)
            cr.stroke()

    # ── komponenty rysowania ──────────────────────────────────────
    def _draw_freq_axis(self, cr, oy, ch):
        cr.set_font_size(9)
        freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        log_min = np.log10(max(self.freq_view_min, 1))
        log_max = np.log10(self.freq_view_max)
        for f in freqs:
            if f < self.freq_view_min or f > self.freq_view_max:
                continue
            pos = (np.log10(f) - log_min) / max(log_max - log_min, 1e-9)
            y = oy + ch - pos * ch
            cr.set_source_rgba(0.55, 0.55, 0.60, 0.7)
            cr.move_to(0, y); cr.line_to(FREQ_W, y)
            cr.set_line_width(0.5); cr.stroke()
            cr.set_source_rgb(0.82, 0.82, 0.82)
            lbl = f"{f}Hz" if f < 1000 else f"{f//1000}kHz"
            cr.move_to(2, y - 2)
            cr.show_text(lbl)

    def _draw_time_ruler(self, cr, ox, oy, cw):
        cr.set_source_rgb(0.16, 0.16, 0.19)
        cr.rectangle(ox, 0, cw, RULER_H)
        cr.fill()
        if len(self.waveform) < 2:
            return
        view_len = max(1, self.view_end - self.view_start)
        n_ticks = max(1, cw // 90)
        cr.set_font_size(9)
        cr.set_source_rgb(0.72, 0.72, 0.72)
        for i in range(n_ticks + 1):
            x = ox + i * cw / n_ticks
            samp = self.view_start + i * view_len / n_ticks
            t_ms = samp / self.sample_rate * 1000
            cr.move_to(x, RULER_H - 7); cr.line_to(x, RULER_H)
            cr.set_line_width(1); cr.stroke()
            cr.move_to(x + 2, RULER_H - 3)
            cr.show_text(f"{t_ms:.0f}ms")

    def _draw_grid(self, cr, ox, oy, cw, ch):
        cr.set_source_rgba(0.24, 0.24, 0.27, 0.45)
        cr.set_line_width(0.5)
        for i in range(1, 4):
            y = oy + i * ch / 4
            cr.move_to(ox, y); cr.line_to(ox + cw, y); cr.stroke()
        for i in range(1, 8):
            x = ox + i * cw / 8
            cr.move_to(x, oy); cr.line_to(x, oy + ch); cr.stroke()

    def _draw_waveform(self, cr, ox, oy, cw, ch):
        cr.set_source_rgba(0.20, 0.72, 0.32, 0.75)
        cr.set_line_width(1.0)
        spx = max(1, self.sample_width_px)
        pts = int(cw // spx)
        if pts < 2:
            return
        view_len = max(1, self.view_end - self.view_start)
        first = True
        for i in range(pts):
            x = ox + i * spx
            idx = int(self.view_start + (i / pts) * view_len)
            if idx >= len(self.waveform):
                break
            y = oy + ch / 2 - self.waveform[idx] * ch / 2
            if first:
                cr.move_to(x, y); first = False
            else:
                cr.line_to(x, y)
        cr.stroke()

    def _draw_segment(self, cr, seg):
        col_sel  = (0.25, 0.82, 1.00)
        col_norm = (1.00, 0.62, 0.10)
        col = col_sel if seg.selected else col_norm

        cr.set_source_rgb(*col)
        cr.set_line_width(2.2 if seg.selected else 1.8)

        x0, y0 = seg.p0
        x1, y1 = seg.p1

        if seg.line_type == LINE_BEZIER:
            cr.move_to(x0, y0)
            cr.curve_to(seg.cp0[0], seg.cp0[1],
                        seg.cp1[0], seg.cp1[1], x1, y1)
            cr.stroke()
            if seg.selected:
                cr.set_source_rgba(0.9, 0.9, 0.25, 0.65)
                cr.set_line_width(1)
                cr.set_dash([4, 4])
                cr.move_to(x0, y0); cr.line_to(seg.cp0[0], seg.cp0[1]); cr.stroke()
                cr.move_to(x1, y1); cr.line_to(seg.cp1[0], seg.cp1[1]); cr.stroke()
                cr.set_dash([])
                for px, py in [seg.cp0, seg.cp1]:
                    cr.set_source_rgba(0.95, 0.90, 0.20, 0.90)
                    cr.arc(px, py, CP_RADIUS - 2, 0, 6.28); cr.fill()

        elif seg.line_type == LINE_LINEAR:
            cr.move_to(x0, y0); cr.line_to(x1, y1); cr.stroke()

        elif seg.line_type == LINE_STEP:
            mx = (x0 + x1) / 2
            cr.move_to(x0, y0)
            cr.line_to(mx, y0)
            cr.line_to(mx, y1)
            cr.line_to(x1, y1)
            cr.stroke()

        # węzły końcowe
        for px, py in [(x0, y0), (x1, y1)]:
            cr.set_source_rgb(*col)
            cr.arc(px, py, CP_RADIUS, 0, 6.28); cr.fill()
            cr.set_source_rgb(0.05, 0.05, 0.05)
            cr.arc(px, py, CP_RADIUS, 0, 6.28)
            cr.set_line_width(1); cr.stroke()

    # ══════════════════════════════════════════════════════════════
    #  Obsługa myszy
    # ══════════════════════════════════════════════════════════════
    def on_button_press(self, widget, event):
        x, y = event.x, event.y
        if event.button == 1:
            if self.current_tool == TOOL_DRAW:
                self.draw_start = (x, y)
            elif self.current_tool == TOOL_EDIT:
                self._edit_press(x, y)
            elif self.current_tool == TOOL_SELECT:
                self._select_press(x, y, event)
        elif event.button == 3:
            self._context_menu(x, y, event)

    def on_button_release(self, widget, event):
        x, y = event.x, event.y
        if event.button == 1:
            if self.current_tool == TOOL_DRAW:
                self._draw_release(x, y)
            elif self.current_tool == TOOL_EDIT:
                self.drag_target = None
            elif self.current_tool == TOOL_SELECT:
                self._select_release(x, y)
        self.drawing_area.queue_draw()
        self._render_waveform_from_segments()
        if self.live_play_active:
            self._live_play_now()

    def on_motion(self, widget, event):
        x, y = event.x, event.y
        self.mouse_pos = (x, y)

        if self.current_tool == TOOL_EDIT and self.drag_target:
            self._edit_move(x, y)
        elif self.current_tool == TOOL_SELECT and self.sel_rect_start:
            self.sel_rect_end = (x, y)
        elif self.current_tool == TOOL_DRAW and self.draw_start:
            pass  # podgląd linii

        self.drawing_area.queue_draw()

        t, a = self._xy_to_normwave(x, y)
        if len(self.waveform) > 0:
            view_len = max(1, self.view_end - self.view_start)
            idx = int(self.view_start + t * view_len)
            ms = max(idx, 0) / self.sample_rate * 1000
            self._status(f"t={ms:.1f}ms  amp={a:.3f}  próbka={max(idx,0)}"
                         f"  |  narzędzie={self.current_tool}  linia={self.current_line}"
                         f"  |  Live={'ON' if self.live_play_active else 'off'}")

    # ── rysowanie segmentu ────────────────────────────────────────
    def _draw_release(self, x, y):
        if self.draw_start is None:
            return
        x0, y0 = self.draw_start
        self.draw_start = None
        if abs(x - x0) < 4 and abs(y - y0) < 4:
            return
        seg = Segment([x0, y0], [x, y], self.current_line)
        self.segments.append(seg)
        self.drawing_area.queue_draw()

    # ── edycja węzłów ─────────────────────────────────────────────
    def _edit_press(self, x, y):
        best = None; best_d = SEL_THRESH ** 2
        for i, seg in enumerate(self.segments):
            for attr in ["p0", "p1", "cp0", "cp1"]:
                if attr in ["cp0", "cp1"] and not seg.selected:
                    continue
                pt = getattr(seg, attr)
                d = (pt[0] - x) ** 2 + (pt[1] - y) ** 2
                if d < best_d:
                    best_d = d; best = (i, attr)
        if best:
            self.drag_target = best
            pt = getattr(self.segments[best[0]], best[1])
            self.drag_offset = (x - pt[0], y - pt[1])

    def _edit_move(self, x, y):
        if not self.drag_target:
            return
        i, attr = self.drag_target
        seg = self.segments[i]
        nx = x - self.drag_offset[0]
        ny = y - self.drag_offset[1]
        pt = getattr(seg, attr)
        dx, dy = nx - pt[0], ny - pt[1]
        pt[0] = nx; pt[1] = ny
        # przesuń uchwyty razem z węzłem końcowym
        if attr == "p0":
            seg.cp0[0] += dx; seg.cp0[1] += dy
        if attr == "p1":
            seg.cp1[0] += dx; seg.cp1[1] += dy

    # ── zaznaczenie ───────────────────────────────────────────────
    def _select_press(self, x, y, event):
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            seg = self._segment_at(x, y)
            if seg:
                seg.selected = not seg.selected
        else:
            for s in self.segments:
                s.selected = False
            self.sel_rect_start = (x, y)
            self.sel_rect_end   = (x, y)
        self.drawing_area.queue_draw()

    def _select_release(self, x, y):
        if self.sel_rect_start:
            x1, y1 = self.sel_rect_start
            rx0, rx1 = min(x1, x), max(x1, x)
            ry0, ry1 = min(y1, y), max(y1, y)
            for seg in self.segments:
                px, py = seg.p0
                if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                    seg.selected = True
                px, py = seg.p1
                if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                    seg.selected = True
            self.sel_rect_start = None
            self.sel_rect_end   = None

    def _segment_at(self, x, y, thresh=14):
        for seg in reversed(self.segments):
            mx = (seg.p0[0] + seg.p1[0]) / 2
            my = (seg.p0[1] + seg.p1[1]) / 2
            if abs(mx - x) < thresh * 3 and abs(my - y) < thresh * 3:
                return seg
        return None

    # ── menu kontekstowe ─────────────────────────────────────────
    def _context_menu(self, x, y, event):
        seg = self._segment_at(x, y)
        menu = Gtk.Menu()
        if seg:
            for lbl, lt in [("Bézier",    LINE_BEZIER),
                             ("Liniowy",   LINE_LINEAR),
                             ("Schodkowy", LINE_STEP)]:
                item = Gtk.MenuItem(label=f"Zmień na: {lbl}")
                item.connect("activate", lambda w, s=seg, t=lt: [setattr(s, 'line_type', t),
                                                                   self.drawing_area.queue_draw()])
                menu.append(item)
            menu.append(Gtk.SeparatorMenuItem())
            item = Gtk.MenuItem(label="Usuń ten segment")
            item.connect("activate", lambda w, s=seg: self._delete_seg(s))
            menu.append(item)
        else:
            item = Gtk.MenuItem(label="Wyczyść wszystko")
            item.connect("activate", self._clear_all)
            menu.append(item)
        menu.show_all()
        menu.popup_at_pointer(event)

    def _delete_seg(self, seg):
        if seg in self.segments:
            self.segments.remove(seg)
        self.drawing_area.queue_draw()
        self._render_waveform_from_segments()

    def _clear_all(self, widget=None):
        self.segments.clear()
        self.waveform = np.array([])
        self.drawing_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Transformacje zaznaczenia
    # ══════════════════════════════════════════════════════════════
    def _transform_selection(self, action):
        sel = [s for s in self.segments if s.selected]
        if not sel:
            self._status("Brak zaznaczonych segmentów!")
            return

        sy = self.sel_scale_y.get_value()
        sx = self.sel_scale_x.get_value()

        # środek zaznaczenia (w px)
        xs = [c for s in sel for c in [s.p0[0], s.p1[0]]]
        ys = [c for s in sel for c in [s.p0[1], s.p1[1]]]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        def scale_pt(pt, scx, scy):
            pt[0] = cx + (pt[0] - cx) * scx
            pt[1] = cy + (pt[1] - cy) * scy

        for seg in sel:
            pts = [seg.p0, seg.p1, seg.cp0, seg.cp1]
            if action == 'compress_y':
                for p in pts: scale_pt(p, 1, 1 / max(sy, 0.01))
            elif action == 'stretch_y':
                for p in pts: scale_pt(p, 1, sy)
            elif action == 'compress_x':
                for p in pts: scale_pt(p, 1 / max(sx, 0.01), 1)
            elif action == 'stretch_x':
                for p in pts: scale_pt(p, sx, 1)
            elif action == 'flip_y':
                for p in pts: p[1] = 2 * cy - p[1]
            elif action == 'flip_x':
                for p in pts: p[0] = 2 * cx - p[0]
            elif action == 'delete':
                if seg in self.segments:
                    self.segments.remove(seg)

        self.drawing_area.queue_draw()
        self._render_waveform_from_segments()
        if self.live_play_active:
            self._live_play_now()

    # ══════════════════════════════════════════════════════════════
    #  Rendering segmentów → waveform (próbkowanie przez Bézier)
    # ══════════════════════════════════════════════════════════════
    def _render_waveform_from_segments(self):
        if not self.segments:
            return
        ox, oy, cw, ch = self._canvas_rect()
        if cw <= 0 or ch <= 0:
            return

        duration = 1.0
        n = int(self.sample_rate * duration)
        wave = np.zeros(n)

        for seg in self.segments:
            t0 = (seg.p0[0] - ox) / cw
            t1 = (seg.p1[0] - ox) / cw
            if t1 < t0:
                # zamień kierunek — zamieniamy węzły
                t0, t1 = t1, t0
                a0 = (oy + ch / 2 - seg.p1[1]) / (ch / 2)
                a1 = (oy + ch / 2 - seg.p0[1]) / (ch / 2)
                cp0a = (oy + ch / 2 - seg.cp1[1]) / (ch / 2)
                cp1a = (oy + ch / 2 - seg.cp0[1]) / (ch / 2)
            else:
                a0   = (oy + ch / 2 - seg.p0[1])  / (ch / 2)
                a1   = (oy + ch / 2 - seg.p1[1])  / (ch / 2)
                cp0a = (oy + ch / 2 - seg.cp0[1]) / (ch / 2)
                cp1a = (oy + ch / 2 - seg.cp1[1]) / (ch / 2)

            i0 = max(0, int(t0 * n))
            i1 = min(n - 1, int(t1 * n))
            if i1 <= i0:
                continue

            ts = np.linspace(0.0, 1.0, i1 - i0)
            if seg.line_type == LINE_BEZIER:
                vals = ((1 - ts) ** 3 * a0
                        + 3 * (1 - ts) ** 2 * ts * cp0a
                        + 3 * (1 - ts) * ts ** 2 * cp1a
                        + ts ** 3 * a1)
            elif seg.line_type == LINE_LINEAR:
                vals = a0 + ts * (a1 - a0)
            elif seg.line_type == LINE_STEP:
                vals = np.where(ts < 0.5, a0, a1)
            else:
                vals = np.zeros_like(ts)

            wave[i0:i1] += vals

        mx = np.max(np.abs(wave))
        if mx > 1e-9:
            wave /= mx

        self.waveform          = wave
        self.original_waveform = wave.copy()
        self.view_start        = 0
        self.view_end          = len(wave)
        self._update_view_range()

    def update_control_points(self):
        pass  # nie potrzebne w nowym modelu

    # ══════════════════════════════════════════════════════════════
    #  Odtwarzanie
    # ══════════════════════════════════════════════════════════════
    def on_play(self, widget=None):
        if len(self.waveform) < 2:
            return
        pygame.mixer.music.load(self._write_tmp_wav(self.waveform))
        pygame.mixer.music.play()

    def on_stop(self, widget=None):
        pygame.mixer.music.stop()

    def _toggle_live_play(self, widget=None):
        self.live_play_active = not self.live_play_active
        self._status("Live play: " + ("ON — każda zmiana od razu odtwarza"
                                      if self.live_play_active else "OFF"))

    def _live_play_now(self):
        if len(self.waveform) < 2:
            return
        pygame.mixer.music.load(self._write_tmp_wav(self.waveform))
        pygame.mixer.music.play()

    def _write_tmp_wav(self, wave):
        path = os.path.join(tempfile.gettempdir(), "ed_waves_live.wav")
        wav.write(path, self.sample_rate, (wave * 32767).astype(np.int16))
        return path

    # ══════════════════════════════════════════════════════════════
    #  Zoom / Scroll
    # ══════════════════════════════════════════════════════════════
    def on_zoom_changed(self, widget):
        self.zoom_level = self.zoom_scale.get_value()
        self._update_view_range()
        self.drawing_area.queue_draw()

    def _update_view_range(self):
        if len(self.waveform) == 0:
            return
        total = len(self.waveform)
        vis   = int(total / max(self.zoom_level, 0.01))
        self.view_start = max(0, min(self.view_start, total - vis))
        self.view_end   = min(total, self.view_start + vis)
        self.vscroll.set_range(0, max(0, total - vis))
        self.vscroll.set_value(self.view_start)

    def on_scroll_changed(self, scrollbar):
        if len(self.waveform) == 0:
            return
        self.view_start = int(scrollbar.get_value())
        vis = int(len(self.waveform) / max(self.zoom_level, 0.01))
        self.view_end = min(len(self.waveform), self.view_start + vis)
        self.drawing_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Zapis / odczyt WAV
    # ══════════════════════════════════════════════════════════════
    def on_save_wav(self, widget):
        dlg = Gtk.FileChooserDialog(title="Zapisz WAV", parent=self,
                                    action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_SAVE,   Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK:
            wav.write(dlg.get_filename(), self.sample_rate,
                      (self.waveform * 32767).astype(np.int16))
        dlg.destroy()

    def on_import_wav(self, widget):
        dlg = Gtk.FileChooserDialog(title="Wczytaj WAV", parent=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OPEN,   Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK:
            self._load_wav(dlg.get_filename())
        dlg.destroy()

    def _load_wav(self, path):
        sr, data = wav.read(path)
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        data = data.astype(float)
        mx = np.max(np.abs(data))
        if mx > 0:
            data /= mx
        self.sample_rate       = sr
        self.waveform          = data
        self.original_waveform = data.copy()
        self.view_start        = 0
        self.view_end          = len(data)
        self._update_view_range()
        self.drawing_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Generacja losowej fali
    # ══════════════════════════════════════════════════════════════
    def on_generate_random_wave(self, widget):
        dlg = Gtk.Dialog(title="Generuj losową falę", parent=self)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OK,     Gtk.ResponseType.OK)
        box = dlg.get_content_area()
        combo = Gtk.ComboBoxText()
        for t in ["lead", "bass", "kick", "tom", "snare"]:
            combo.append_text(t)
        combo.set_active(0)
        box.pack_start(combo, False, False, 4)
        dur = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        dur.set_range(0.1, 3.0); dur.set_value(1.0); dur.set_digits(1)
        box.pack_start(dur, False, False, 4)
        box.show_all()
        if dlg.run() == Gtk.ResponseType.OK:
            self._gen_random(combo.get_active_text(), dur.get_value())
        dlg.destroy()

    def _gen_random(self, wtype, duration):
        n = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        if wtype == "lead":
            f = random.uniform(220, 880)
            w = (np.sin(2*np.pi*f*t) + .5*np.sin(4*np.pi*f*t) + .25*np.sin(6*np.pi*f*t))
        elif wtype == "bass":
            f = random.uniform(41, 165)
            w = np.tanh(np.sin(2*np.pi*f*t) * 2) / 2
        elif wtype == "kick":
            f = random.uniform(50, 100)
            w = np.sin(2*np.pi*f*np.exp(-t*10)*t)
        elif wtype == "tom":
            f = random.uniform(100, 300)
            w = np.sin(2*np.pi*f*np.exp(-t*5)*t)
        else:
            w = (.5*np.random.normal(0, 1, n) + .5*np.sin(2*np.pi*180*t)) * np.exp(-t*20)
        mx = np.max(np.abs(w))
        if mx > 0:
            w /= mx
        self.waveform          = w
        self.original_waveform = w.copy()
        self.view_start        = 0
        self.view_end          = len(w)
        self._update_view_range()
        self.drawing_area.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Statusbar
    # ══════════════════════════════════════════════════════════════
    def _status(self, msg):
        self.statusbar.set_text(msg)


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = WaveformEditor()
    app.show_all()
    Gtk.main()
