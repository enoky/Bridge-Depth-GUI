#!/usr/bin/env python

import sys
import os
import glob
import threading
import json

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

# Assuming the 'bridge', 'depth_scaler', and 'dilation' files are in your project's path
from bridge.dpt import Bridge
from depth_scaler import EMAMinMaxScaler
from dilation import dilate_edge

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
                    
                    normalized_tensor = None
                    if scaler:
                        depth_tensor = torch.from_numpy(depth_numpy).to(self.settings['device'])
                        normalized_tensor = scaler(depth_tensor)
                        if normalized_tensor is None:
                            self.progress.emit(i + 1)
                            continue 
                        normalized_tensor = 1.0 - normalized_tensor # Re-invert
                    else:
                        min_val, max_val = np.min(depth_numpy), np.max(depth_numpy)
                        normalized = (depth_numpy - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(depth_numpy)
                        normalized_tensor = torch.from_numpy(normalized).to(self.settings['device'])

                    # --- Post-processing and saving ---
                    final_tensor = self._apply_post_processing(normalized_tensor)
                    depth_uint8 = (final_tensor.cpu().numpy() * 255).astype(np.uint8)
                    output_frame = self._colorize_frame(depth_uint8)
                    writer.write(output_frame)
                    self.progress.emit(i + 1)
            
            if scaler:
                self.log.emit("ℹ️ Flushing temporal buffer...")
                for normalized_tensor in scaler.flush():
                    normalized_tensor = 1.0 - normalized_tensor # Re-invert
                    final_tensor = self._apply_post_processing(normalized_tensor)
                    depth_uint8 = (final_tensor.cpu().numpy() * 255).astype(np.uint8)
                    output_frame = self._colorize_frame(depth_uint8)
                    writer.write(output_frame)
            
            if self.is_running:
                self.log.emit(f"✅ Video saved to '{output_path}'")

        finally:
            if cap: cap.release()
            if writer: writer.release()
            
    def _get_output_resolution(self, width, height, filename):
        output_width, output_height = width, height
        max_res = self.settings['max_width']
        if max_res > 0 and width > max_res:
            self.log.emit(f"ℹ️ Resizing '{filename}' from {width}x{height} to {max_res}p width.")
            ratio = max_res / width
            output_width = max_res
            output_height = int(height * ratio)
        return output_width, output_height

    def _apply_post_processing(self, tensor):
        """Applies optional post-processing effects like edge dilation."""
        if self.settings['use_edge_dilation']:
            # dilate_edge expects a 4D tensor (B, C, H, W)
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
            dilated_tensor = dilate_edge(tensor_4d, n=self.settings['dilation_iterations'])
            return dilated_tensor.squeeze(0).squeeze(0)
        return tensor

    def _colorize_frame(self, depth_uint8):
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
        self.setGeometry(100, 100, 750, 650)
        self.config_file = "config.json"

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

        self.load_settings()

    def _create_folder_selectors(self):
        self.layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_path = self._create_folder_selector_widget()
        self.layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_path = self._create_folder_selector_widget()

    def _create_options_ui(self):
        # --- Main Options Container ---
        main_options_layout = QHBoxLayout()
        self.layout.addLayout(main_options_layout)
        
        # --- Left Side: General and Temporal ---
        left_container = QFrame()
        left_container.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_container)
        main_options_layout.addWidget(left_container)

        left_layout.addWidget(QLabel("<b>General Options</b>"))
        # ... Colormap and Max Width ...
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['none', 'inferno', 'viridis', 'magma', 'jet', 'plasma'])
        colormap_layout.addWidget(self.colormap_combo)
        left_layout.addLayout(colormap_layout)
        
        max_width_layout = QHBoxLayout()
        max_width_layout.addWidget(QLabel("Max Width:"))
        self.max_res_input = QLineEdit()
        self.max_res_input.setPlaceholderText("e.g., 1024 (optional)")
        max_width_layout.addWidget(self.max_res_input)
        left_layout.addLayout(max_width_layout)

        left_layout.addWidget(self._create_separator())
        
        left_layout.addWidget(QLabel("<b>Temporal Smoothing</b>"))
        self.temporal_checkbox = QCheckBox("Enable")
        self.temporal_checkbox.setChecked(True)
        left_layout.addWidget(self.temporal_checkbox)

        ema_layout = QHBoxLayout()
        ema_layout.addWidget(QLabel("Decay (EMA):"))
        self.ema_decay_input = QLineEdit("0.99")
        ema_layout.addWidget(self.ema_decay_input)
        left_layout.addLayout(ema_layout)

        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer Size:"))
        self.ema_buffer_input = QLineEdit("30")
        buffer_layout.addWidget(self.ema_buffer_input)
        left_layout.addLayout(buffer_layout)
        left_layout.addStretch()

        # --- NEW: Right Side for Post-Processing ---
        right_container = QFrame()
        right_container.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_container)
        main_options_layout.addWidget(right_container)

        right_layout.addWidget(QLabel("<b>Post-Processing</b>"))
        self.edge_dilation_checkbox = QCheckBox("Enable Edge Dilation")
        self.edge_dilation_checkbox.setChecked(False)
        right_layout.addWidget(self.edge_dilation_checkbox)

        dilation_iter_layout = QHBoxLayout()
        dilation_iter_layout.addWidget(QLabel("Iterations:"))
        self.dilation_iter_input = QLineEdit("1")
        dilation_iter_layout.addWidget(self.dilation_iter_input)
        right_layout.addLayout(dilation_iter_layout)
        right_layout.addStretch()

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

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
        if not settings: return

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
            'use_temporal_smoothing': self.temporal_checkbox.isChecked(),
            'use_edge_dilation': self.edge_dilation_checkbox.isChecked(),
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
            if settings['use_edge_dilation']:
                settings['dilation_iterations'] = int(self.dilation_iter_input.text())
        except ValueError:
            self.log_box.append("❌ Invalid number in one of the settings fields.")
            return None
            
        return settings

    def load_settings(self):
        """Loads settings from config.json on startup."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.input_folder_path.setText(config.get("input_folder", ""))
                self.output_folder_path.setText(config.get("output_folder", ""))
                self.colormap_combo.setCurrentText(config.get("colormap", "none"))
                self.max_res_input.setText(config.get("max_width", ""))
                self.temporal_checkbox.setChecked(config.get("use_temporal_smoothing", True))
                self.ema_decay_input.setText(config.get("ema_decay", "0.99"))
                self.ema_buffer_input.setText(config.get("ema_buffer", "30"))
                self.edge_dilation_checkbox.setChecked(config.get("use_edge_dilation", False))
                self.dilation_iter_input.setText(config.get("dilation_iterations", "1"))
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Could not load config file: {e}")

    def save_settings(self):
        """Saves current UI settings to config.json on exit."""
        config = {
            "input_folder": self.input_folder_path.text(),
            "output_folder": self.output_folder_path.text(),
            "colormap": self.colormap_combo.currentText(),
            "max_width": self.max_res_input.text(),
            "use_temporal_smoothing": self.temporal_checkbox.isChecked(),
            "ema_decay": self.ema_decay_input.text(),
            "ema_buffer": self.ema_buffer_input.text(),
            "use_edge_dilation": self.edge_dilation_checkbox.isChecked(),
            "dilation_iterations": self.dilation_iter_input.text(),
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Could not save config file: {e}")

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
        self.save_settings()
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
