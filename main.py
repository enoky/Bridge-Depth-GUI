#!/usr/bin/env python

import sys
import os
import glob
import threading

# --- GUI ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QComboBox, QCheckBox, QFrame
)
from PySide6.QtCore import QObject, Signal, QThread, Qt

# --- Processing ---
import cv2
import torch
import numpy as np

# Assuming the 'bridge' and 'depth_scaler' files are in your project's path
from bridge.dpt import Bridge
from depth_scaler import EMAMinMaxScaler

# --- Worker for background processing ---
class ProcessingWorker(QObject):
    """
    Runs the video processing task on a separate thread to keep the GUI responsive.
    """
    # Signals to communicate with the main GUI thread
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    video_started = Signal(str, int) # video_name, total_frames

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.is_running = True

    def run(self):
        """Main processing loop."""
        try:
            self.log.emit("✅ Starting batch processing...")
            
            # 1. --- Model Loading ---
            self.log.emit(f"⏳ Loading model from '{self.settings['model_path']}'...")
            model = Bridge()
            model.load_state_dict(torch.load(self.settings['model_path'], map_location=self.settings['device']))
            model = model.to(self.settings['device']).eval()
            self.log.emit("✅ Model loaded successfully.")

            # 2. --- Find Videos ---
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.settings['input_folder'], ext)))

            if not video_files:
                self.log.emit("❌ No video files found in the input folder.")
                self.finished.emit()
                return
            
            self.log.emit(f"Found {len(video_files)} video(s) to process.")

            # 3. --- Process Each Video ---
            for video_path in video_files:
                if not self.is_running:
                    break
                basename = os.path.basename(video_path)
                filename, _ = os.path.splitext(basename)
                output_filename = f"{filename}_depth.mp4"
                output_path = os.path.join(self.settings['output_folder'], output_filename)
                
                self._process_single_video(video_path, output_path, model)
            
            self.log.emit("\n🎉 All videos processed.")

        except Exception as e:
            self.log.emit(f"❌ An unexpected error occurred: {e}")
        finally:
            self.finished.emit()

    def _process_single_video(self, video_path, output_path, model):
        """Processes one video file frame by frame."""
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log.emit(f"❌ Error: Could not open video file {os.path.basename(video_path)}")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_width, output_height = self._get_output_resolution(frame_width, frame_height, os.path.basename(video_path))
            
            self.video_started.emit(os.path.basename(video_path), total_frames)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

            # --- NORMALIZATION SETUP ---
            scaler = None
            if self.settings['use_temporal_smoothing']:
                self.log.emit(f"ℹ️ Using Temporal Smoothing (Decay: {self.settings['ema_decay']}, Buffer: {self.settings['ema_buffer']})")
                scaler = EMAMinMaxScaler(decay=self.settings['ema_decay'], buffer_size=self.settings['ema_buffer'])

            with torch.no_grad():
                for i in range(total_frames):
                    if not self.is_running:
                        self.log.emit("🛑 Processing stopped by user.")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if (frame_width, frame_height) != (output_width, output_height):
                        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

                    depth_numpy = model.infer_image(frame).squeeze()
                    
                    # --- NORMALIZATION LOGIC ---
                    if scaler:
                        depth_tensor = torch.from_numpy(depth_numpy).to(self.settings['device'])
                        normalized_tensor = scaler(depth_tensor)
                        if normalized_tensor is None: # Buffer not full yet
                            self.progress.emit(i + 1)
                            continue 
                        # The scaler inverts depth (close=white). We re-invert it here for consistency (close=black).
                        depth_uint8 = ((1.0 - normalized_tensor).cpu().numpy() * 255).astype(np.uint8)
                    else: # Per-frame normalization
                        min_val, max_val = np.min(depth_numpy), np.max(depth_numpy)
                        normalized_depth = (depth_numpy - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(depth_numpy)
                        depth_uint8 = (normalized_depth * 255).astype(np.uint8)

                    output_frame = self._colorize_frame(depth_uint8)
                    writer.write(output_frame)
                    self.progress.emit(i + 1)
            
            # --- FLUSH SCALER BUFFER ---
            if scaler:
                self.log.emit("ℹ️ Flushing temporal buffer...")
                flushed_frames = scaler.flush()
                for normalized_tensor in flushed_frames:
                    # Re-invert the flushed frames as well
                    depth_uint8 = ((1.0 - normalized_tensor).cpu().numpy() * 255).astype(np.uint8)
                    output_frame = self._colorize_frame(depth_uint8)
                    writer.write(output_frame)
            
            if self.is_running:
                self.log.emit(f"✅ Video saved to '{output_path}'")

        finally:
            if cap: cap.release()
            if writer: writer.release()
            
    def _get_output_resolution(self, width, height, filename):
        """Calculates and logs the final output resolution."""
        output_width, output_height = width, height
        max_res = self.settings['max_width']
        if max_res > 0 and width > max_res:
            self.log.emit(f"ℹ️ Resizing '{filename}' from {width}x{height} to {max_res}p width.")
            ratio = max_res / width
            output_width = max_res
            output_height = int(height * ratio)
        return output_width, output_height

    def _colorize_frame(self, depth_uint8):
        """Applies colormap or converts to BGR based on settings."""
        if self.settings['colormap'] != 'none':
            colormap_func = getattr(cv2, f"COLORMAP_{self.settings['colormap'].upper()}", cv2.COLORMAP_INFERNO)
            return cv2.applyColorMap(depth_uint8, colormap_func)
        else:
            return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    def stop(self):
        self.is_running = False

# --- Main GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Depth Estimator")
        self.setGeometry(100, 100, 700, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self._create_folder_selectors()
        self._create_options_ui()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self._create_progress_and_log_ui()
        
        self.thread = None
        self.worker = None

    def _create_folder_selectors(self):
        self.layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_path = self._create_folder_selector_widget()
        self.layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_path = self._create_folder_selector_widget()

    def _create_options_ui(self):
        options_layout = QHBoxLayout()
        # Colormap and Max Width
        left_opts_layout = QVBoxLayout()
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['none', 'inferno', 'viridis', 'magma', 'jet', 'plasma'])
        colormap_layout.addWidget(self.colormap_combo)
        left_opts_layout.addLayout(colormap_layout)
        
        max_width_layout = QHBoxLayout()
        max_width_layout.addWidget(QLabel("Max Width:"))
        self.max_res_input = QLineEdit()
        self.max_res_input.setPlaceholderText("e.g., 1024 (optional)")
        max_width_layout.addWidget(self.max_res_input)
        left_opts_layout.addLayout(max_width_layout)
        options_layout.addLayout(left_opts_layout)

        # --- NEW: Temporal Smoothing Options ---
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        options_layout.addWidget(line)

        right_opts_layout = QVBoxLayout()
        self.temporal_checkbox = QCheckBox("Enable Temporal Smoothing")
        self.temporal_checkbox.setChecked(True)
        right_opts_layout.addWidget(self.temporal_checkbox)

        ema_layout = QHBoxLayout()
        ema_layout.addWidget(QLabel("Decay (EMA):"))
        self.ema_decay_input = QLineEdit("0.9")
        ema_layout.addWidget(self.ema_decay_input)
        right_opts_layout.addLayout(ema_layout)

        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer Size:"))
        self.ema_buffer_input = QLineEdit("30")
        buffer_layout.addWidget(self.ema_buffer_input)
        right_opts_layout.addLayout(buffer_layout)

        options_layout.addLayout(right_opts_layout)
        self.layout.addLayout(options_layout)

    def _create_progress_and_log_ui(self):
        self.progress_label = QLabel("Waiting to start...")
        self.layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)

    def _create_folder_selector_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)
        line_edit = QLineEdit()
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._select_directory(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        self.layout.addWidget(widget)
        return line_edit
    
    def _select_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if dir_path: line_edit.setText(dir_path)

    def start_processing(self):
        settings = self.get_settings()
        if not settings:
            return

        os.makedirs(settings['output_folder'], exist_ok=True)
        self.start_button.setEnabled(False)
        self.start_button.setText("Processing...")
        
        self.log_box.append(f"✅ Using device: {settings['device']}")

        self.thread = QThread()
        self.worker = ProcessingWorker(settings)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.on_processing_finished)
        
        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.video_started.connect(self.on_video_started)

        self.thread.start()

    def get_settings(self):
        """Gathers and validates all UI settings."""
        settings = {
            'input_folder': self.input_folder_path.text(),
            'output_folder': self.output_folder_path.text(),
            'model_path': "checkpoints/bridge.pth",
            'colormap': self.colormap_combo.currentText(),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'use_temporal_smoothing': self.temporal_checkbox.isChecked()
        }
        
        if not all([settings['input_folder'], settings['output_folder']]):
            self.log_box.append("❌ Please select the input and output folders.")
            return None

        if not os.path.exists(settings['model_path']):
            self.log_box.append(f"❌ Error: Model not found at '{settings['model_path']}'")
            return None

        try:
            settings['max_width'] = int(self.max_res_input.text()) if self.max_res_input.text() else 0
            if settings['use_temporal_smoothing']:
                settings['ema_decay'] = float(self.ema_decay_input.text())
                settings['ema_buffer'] = int(self.ema_buffer_input.text())
        except ValueError:
            self.log_box.append("❌ Invalid number in 'Max Width', 'Decay', or 'Buffer Size'.")
            return None
            
        return settings

    def on_video_started(self, name, total_frames):
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Processing: {name}")

    def on_processing_finished(self):
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Processing")
        self.progress_label.setText("Finished!")
        self.progress_bar.setValue(0)
        self.thread = None
        self.worker = None
        
    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
