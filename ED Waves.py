import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import cairo
import numpy as np
import pygame
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
import random

class WaveformEditor(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Waveform Editor")
        self.set_default_size(1200, 600)

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(self.box)

        self.scroll_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.box.pack_start(self.scroll_container, True, True, 0)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(1100, 400)
        self.drawing_area.connect("draw", self.on_draw)
        self.drawing_area.connect("button-press-event", self.on_button_press)
        self.drawing_area.connect("button-release-event", self.on_button_release)
        self.drawing_area.connect("motion-notify-event", self.on_motion)
        self.drawing_area.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                                     Gdk.EventMask.BUTTON_RELEASE_MASK |
                                     Gdk.EventMask.POINTER_MOTION_MASK)
        self.scroll_container.pack_start(self.drawing_area, True, True, 0)

        self.scrollbar = Gtk.VScrollbar()
        self.scrollbar.connect("value-changed", self.on_scroll_changed)
        self.scroll_container.pack_start(self.scrollbar, False, False, 0)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.box.pack_start(button_box, False, False, 0)

        self.edit_button = Gtk.ToggleButton(label="Edit")
        self.edit_button.connect("toggled", self.on_tool_toggled)
        button_box.pack_start(self.edit_button, False, False, 0)

        play_button = Gtk.Button(label="Play")
        play_button.connect("clicked", self.on_play)
        button_box.pack_start(play_button, False, False, 0)

        save_button = Gtk.Button(label="Save WAV")
        save_button.connect("clicked", self.on_save_wav)
        button_box.pack_start(save_button, False, False, 0)

        import_button = Gtk.Button(label="Import WAV")
        import_button.connect("clicked", self.on_import_wav)
        button_box.pack_start(import_button, False, False, 0)

        generate_button = Gtk.Button(label="Generate Random Wave")
        generate_button.connect("clicked", self.on_generate_random_wave)
        button_box.pack_start(generate_button, False, False, 0)

        instrument_button = Gtk.Button(label="Select Instrument")
        instrument_button.connect("clicked", self.on_select_instrument)
        button_box.pack_start(instrument_button, False, False, 0)

        self.zoom_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.zoom_scale.set_range(1, 10)
        self.zoom_scale.set_value(1)
        self.zoom_scale.set_digits(1)
        self.zoom_scale.connect("value-changed", self.on_zoom_changed)
        self.zoom_scale.set_increments(0.1, 1)
        self.zoom_scale.set_hexpand(True)
        self.box.pack_start(self.zoom_scale, False, False, 0)

        self.control_points = []
        self.waveform = np.array([])
        self.original_waveform = np.array([])
        self.current_tool = "edit"
        self.dragging_point = None
        self.zoom_level = 1
        self.view_start = 0
        self.view_end = 0
        self.sample_rate = 44100

        self.instruments_db = self.load_instruments_db()

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=4096)

    def load_instruments_db(self):
        instruments = {}
        with open('instr_db.txt', 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(';')
                name = parts[0]
                instruments[name] = {
                    'family': parts[1],
                    'pitch_range': [float(parts[2]), float(parts[3])],
                    'attack': float(parts[4]),
                    'decay': float(parts[5]),
                    'sustain': float(parts[6]),
                    'release': float(parts[7]),
                    'special1': parts[8],
                    'special2': float(parts[9]),
                    'default_params': [float(p) for p in parts[10:14]]
                }
        return instruments

    def on_select_instrument(self, widget):
        dialog = Gtk.Dialog(title="Select Instrument", parent=self)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OK, Gtk.ResponseType.OK)

        box = dialog.get_content_area()

        instrument_combo = Gtk.ComboBoxText()
        for instrument in sorted(self.instruments_db.keys()):
            instrument_combo.append_text(instrument)
        instrument_combo.set_active(0)
        box.pack_start(instrument_combo, False, False, 0)

        parameter_adjustments = {}
        params = ['attack', 'decay', 'sustain', 'release', 'special1', 'special2']
        for i, param in enumerate(params):
            label = Gtk.Label(label=param)
            box.pack_start(label, False, False, 0)
            adjustment = Gtk.Adjustment(value=0, lower=0, upper=1, step_increment=0.01, page_increment=0.1, page_size=0)
            scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adjustment)
            scale.set_digits(2)
            box.pack_start(scale, False, False, 0)
            parameter_adjustments[param] = scale

        def update_default_values(combo):
            instrument = combo.get_active_text()
            default_params = self.instruments_db[instrument]['default_params']
            for i, (param, scale) in enumerate(parameter_adjustments.items()):
                if i < len(default_params):
                    scale.set_value(default_params[i])

        instrument_combo.connect('changed', update_default_values)
        update_default_values(instrument_combo)

        box.show_all()

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            selected_instrument = instrument_combo.get_active_text()
            parameters = {param: scale.get_value() for param, scale in parameter_adjustments.items()}
            self.generate_instrument_wave(selected_instrument, parameters)

        dialog.destroy()

    def find_nearest_point(self, x, y):
        if not self.control_points:
            return None
        distances = [(i, (p[0]-x)**2 + (p[1]-y)**2) for i, p in enumerate(self.control_points)]
        nearest = min(distances, key=lambda d: d[1])
        return nearest[0] if nearest[1] < 100 else None

    def update_waveform(self, edited_point_index):
        if len(self.control_points) < 2:
            return

        width = self.drawing_area.get_allocated_width()
        height = self.drawing_area.get_allocated_height()

        x_points = [int(self.view_start + (p[0] / width) * (self.view_end - self.view_start)) for p in self.control_points]
        y_points = [(height / 2 - p[1]) / (height / 2) for p in self.control_points]

        influence_range = 1000
        start_index = max(0, x_points[edited_point_index] - influence_range)
        end_index = min(len(self.waveform), x_points[edited_point_index] + influence_range)

        local_x_points = [x for x in x_points if start_index <= x <= end_index]
        local_y_points = [y_points[x_points.index(x)] for x in local_x_points]

        if len(local_x_points) > 1:
            f = interp1d(local_x_points, local_y_points, kind='cubic', fill_value='extrapolate')
            local_indices = np.arange(start_index, end_index)
            new_local_waveform = f(local_indices)

            transition = np.linspace(0, 1, influence_range)
            new_local_waveform[:influence_range] = (transition * new_local_waveform[:influence_range] +
                                                    (1 - transition) * self.waveform[start_index:start_index+influence_range])
            new_local_waveform[-influence_range:] = (transition[::-1] * new_local_waveform[-influence_range:] +
                                                     (1 - transition[::-1]) * self.waveform[end_index-influence_range:end_index])

            self.waveform[start_index:end_index] = new_local_waveform

    def generate_instrument_wave(self, instrument, parameters):
        instrument_data = self.instruments_db[instrument]

        # Base frequency setup
        base_frequency = np.mean(instrument_data['pitch_range'])
        duration = max(1.0, parameters['attack'] + parameters['decay'] + parameters['release'])
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # Generate base waveform
        waveform = np.sin(2 * np.pi * base_frequency * t)

        # Add harmonics
        harmonic_amplitudes = [1.0, 0.5, 0.25]  # Example harmonics (can be made editable)
        waveform += sum(harmonic_amplitudes[n] * np.sin(2 * np.pi * (n+2) * base_frequency * t) for n in range(len(harmonic_amplitudes)))

        # Apply ADSR
        envelope = self.apply_adsr(parameters, len(t))
        waveform *= envelope

        # Apply instrument-specific effects
        waveform = self.apply_instrument_effects(instrument_data, parameters, waveform, t)

        # Normalize waveform
        waveform /= np.max(np.abs(waveform))

        self.waveform = waveform
        self.original_waveform = waveform.copy()
        self.view_start = 0
        self.view_end = len(self.waveform)
        self.update_control_points()
        self.drawing_area.queue_draw()

    def apply_adsr(self, parameters, num_samples):
        t = np.linspace(0, 1, num_samples)
        attack_samples = int(parameters['attack'] * self.sample_rate)
        decay_samples = int(parameters['decay'] * self.sample_rate)
        release_samples = int(parameters['release'] * self.sample_rate)

        sustain_level = parameters['sustain']
        envelope = np.ones_like(t)

        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        decay_end = attack_samples + decay_samples
        envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)
        sustain_end = len(t) - release_samples
        envelope[decay_end:sustain_end] = sustain_level
        envelope[sustain_end:] = np.linspace(sustain_level, 0, release_samples)

        return envelope

    def apply_instrument_effects(self, instrument_data, parameters, waveform, t):
        family = instrument_data['family']
        if family == 'string':
            vibrato_rate = parameters['special1']
            vibrato_depth = parameters['special2']
            vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            waveform *= (1 + vibrato)
        elif family == 'wind':
            breath_noise = parameters['special1'] * np.random.normal(0, 1, len(t))
            vibrato = parameters['special2'] * np.sin(2 * np.pi * 5 * t)
            waveform += breath_noise
            waveform *= (1 + vibrato)
        elif family == 'percussion':
            resonance = parameters['special1']
            damping = parameters['special2']
            waveform *= np.exp(-damping * t)
        elif family == 'electronic':
            wave_shape = parameters['special1']
            if wave_shape == 'square':
                waveform = np.sign(waveform)
            elif wave_shape == 'sawtooth':
                waveform = 2 * (t * np.mean(instrument_data['pitch_range']) - np.floor(0.5 + t * np.mean(instrument_data['pitch_range'])))
            filter_cutoff = parameters['special2']
            waveform = self.apply_lowpass_filter(waveform, filter_cutoff)
        return waveform

    def apply_lowpass_filter(self, waveform, cutoff_freq):
        # Implement a simple low-pass filter (this can be replaced with a better one)
        return waveform  # For simplicity, returning the waveform without filtering

    def on_play(self, widget=None):
        pygame.mixer.music.load(self.create_temp_wav(self.waveform))
        pygame.mixer.music.play()

    def create_temp_wav(self, waveform):
        temp_wav_path = "temp_waveform.wav"
        wav.write(temp_wav_path, 44100, (waveform * 32767).astype(np.int16))
        return temp_wav_path

    def on_save_wav(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Save WAV File", parent=self, action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )

        filter_wav = Gtk.FileFilter()
        filter_wav.set_name("WAV Files")
        filter_wav.add_mime_type("audio/wav")
        dialog.add_filter(filter_wav)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            file_path = dialog.get_filename()
            self.save_wav(file_path)

        dialog.destroy()

    def save_wav(self, file_path):
        wav.write(file_path, 44100, (self.waveform * 32767).astype(np.int16))

    def on_import_wav(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Choose WAV File", parent=self, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        filter_wav = Gtk.FileFilter()
        filter_wav.set_name("WAV Files")
        filter_wav.add_mime_type("audio/wav")
        dialog.add_filter(filter_wav)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            file_path = dialog.get_filename()
            self.load_wav(file_path)

        dialog.destroy()

    def on_draw(self, widget, cr):
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        width = self.drawing_area.get_allocated_width()
        height = self.drawing_area.get_allocated_height()

        cr.set_source_rgb(0.7, 0.7, 0.7)
        cr.move_to(0, height / 2)
        cr.line_to(width, height / 2)
        cr.stroke()

        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(2)

        if len(self.waveform) > 1:
            cr.move_to(0, height / 2)
            for i in range(width):
                x = i
                index = int(self.view_start + (i / width) * (self.view_end - self.view_start))
                if index < len(self.waveform):
                    y = height / 2 - (self.waveform[index] * height / 2)
                    cr.line_to(x, y)
            cr.stroke()

        for x, y in self.control_points:
            cr.arc(x, y, 5, 0, 2 * np.pi)
            cr.fill()

    def load_wav(self, file_path):
        self.sample_rate, audio_data = wav.read(file_path)
        if len(audio_data.shape) == 2:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(float)
        audio_data /= np.max(np.abs(audio_data))
        self.waveform = audio_data
        self.original_waveform = audio_data.copy()
        self.view_start = 0
        self.view_end = len(self.waveform)
        self.update_control_points()
        self.drawing_area.queue_draw()

    def on_zoom_changed(self, widget):
        self.zoom_level = self.zoom_scale.get_value()
        self.update_view_range()
        self.update_control_points()
        self.drawing_area.queue_draw()

    def update_view_range(self):
        total_samples = len(self.waveform)
        visible_samples = int(total_samples / self.zoom_level)
        self.view_start = max(0, min(self.view_start, total_samples - visible_samples))
        self.view_end = min(total_samples, self.view_start + visible_samples)
        self.scrollbar.set_range(0, total_samples - visible_samples)
        self.scrollbar.set_value(self.view_start)

    def update_control_points(self):
        width = self.drawing_area.get_allocated_width()
        height = self.drawing_area.get_allocated_height()
        self.control_points = []
        num_points = 40
        for i in range(num_points):
            x = int(i * width / (num_points - 1))
            index = int(self.view_start + (x / width) * (self.view_end - self.view_start))
            if index < len(self.waveform):
                y = height / 2 - (self.waveform[index] * height / 2)
                self.control_points.append((x, y))

    def on_scroll_changed(self, scrollbar):
        self.view_start = int(scrollbar.get_value())
        self.view_end = min(len(self.waveform), self.view_start + int(len(self.waveform) / self.zoom_level))
        self.update_control_points()
        self.drawing_area.queue_draw()

    def on_tool_toggled(self, button):
        self.current_tool = "edit" if button.get_active() else None

    def on_button_press(self, widget, event):
        if self.current_tool == "edit":
            self.dragging_point = self.find_nearest_point(event.x, event.y)

    def on_button_release(self, widget, event):
        if self.current_tool == "edit" and self.dragging_point is not None:
            self.update_waveform(self.dragging_point)
            self.dragging_point = None
        self.drawing_area.queue_draw()

    def on_motion(self, widget, event):
        if self.current_tool == "edit" and self.dragging_point is not None:
            x, y = event.x, event.y
            self.control_points[self.dragging_point] = (x, y)
            self.drawing_area.queue_draw()

    def on_generate_random_wave(self, widget):
        dialog = Gtk.Dialog(title="Generate Random Wave", parent=self)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OK, Gtk.ResponseType.OK)

        box = dialog.get_content_area()

        wave_types = ["lead", "bass", "kick", "tom", "snare"]
        wave_type_combo = Gtk.ComboBoxText()
        for wave_type in wave_types:
            wave_type_combo.append_text(wave_type)
        wave_type_combo.set_active(0)
        box.pack_start(wave_type_combo, False, False, 0)

        duration_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        duration_scale.set_range(0.1, 3.0)
        duration_scale.set_value(1.0)
        duration_scale.set_digits(1)
        duration_scale.set_increments(0.1, 0.5)
        box.pack_start(duration_scale, False, False, 0)

        box.show_all()

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            wave_type = wave_type_combo.get_active_text()
            duration = duration_scale.get_value()
            self.generate_random_wave(wave_type, duration)

        dialog.destroy()

    def generate_random_wave(self, wave_type, duration):
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        if wave_type == "lead":
            frequency = random.uniform(220, 880)  # A3 to A5
            waveform = np.sin(2 * np.pi * frequency * t)
            waveform += 0.5 * np.sin(4 * np.pi * frequency * t)
            waveform += 0.25 * np.sin(6 * np.pi * frequency * t)
        elif wave_type == "bass":
            frequency = random.uniform(41, 165)  # E1 to E3
            waveform = np.sin(2 * np.pi * frequency * t)
            waveform = np.tanh(waveform * 2) / 2
        elif wave_type == "kick":
            frequency = random.uniform(50, 100)
            waveform = np.sin(2 * np.pi * frequency * np.exp(-t * 10) * t)
        elif wave_type == "tom":
            frequency = random.uniform(100, 300)
            waveform = np.sin(2 * np.pi * frequency * np.exp(-t * 5) * t)
        elif wave_type == "snare":
            noise = np.random.normal(0, 1, num_samples)
            tone = np.sin(2 * np.pi * 180 * t)
            waveform = 0.5 * noise + 0.5 * tone
            waveform *= np.exp(-t * 20)

        waveform /= np.max(np.abs(waveform))

        self.waveform = waveform
        self.original_waveform = waveform.copy()
        self.view_start = 0
        self.view_end = len(self.waveform)
        self.update_control_points()
        self.drawing_area.queue_draw()

if __name__ == "__main__":
    app = WaveformEditor()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()
