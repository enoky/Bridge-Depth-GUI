import sys
import os
import glob
import threading

# --- GUI ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QComboBox
)
from PySide6.QtCore import QObject, Signal, QThread, Qt

# --- Processing ---
import cv2
import torch
import numpy as np

# Assuming the 'bridge' directory is in your project's path
from bridge.dpt import Bridge

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

    def __init__(self, input_folder, output_folder, model_path, colormap, device, max_resolution_str):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        self.colormap = colormap
        self.device = device
        self.max_resolution_str = max_resolution_str
        self.is_running = True

    def run(self):
        """Main processing loop."""
        try:
            self.log.emit("✅ Starting batch processing...")
            
            # 1. --- Model Loading ---
            self.log.emit(f"⏳ Loading model from '{self.model_path}'...")
            # Path existence is already checked in the main thread
            
            model = Bridge()
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device).eval()
            self.log.emit("✅ Model loaded successfully.")

            # 2. --- Find Videos ---
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.input_folder, ext)))

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
                output_path = os.path.join(self.output_folder, output_filename)
                
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

            # Original video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine output resolution
            output_width, output_height = frame_width, frame_height
            max_res = 0
            try:
                if self.max_resolution_str:
                    max_res = int(self.max_resolution_str)
            except ValueError:
                self.log.emit(f"⚠️ Invalid max width value '{self.max_resolution_str}'. Ignoring.")

            if max_res > 0 and frame_width > max_res:
                self.log.emit(f"ℹ️ Resizing '{os.path.basename(video_path)}' from {frame_width}x{frame_height} to {max_res}p width.")
                ratio = max_res / frame_width
                output_width = max_res
                output_height = int(frame_height * ratio)

            self.video_started.emit(os.path.basename(video_path), total_frames)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

            with torch.no_grad():
                for i in range(total_frames):
                    if not self.is_running:
                        self.log.emit("🛑 Processing stopped by user.")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize frame if necessary before processing
                    if (frame_width, frame_height) != (output_width, output_height):
                        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

                    depth = model.infer_image(frame)
                    depth_numpy = depth.squeeze()

                    min_val, max_val = np.min(depth_numpy), np.max(depth_numpy)
                    normalized_depth = (depth_numpy - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(depth_numpy)
                    depth_uint8 = (normalized_depth * 255).astype(np.uint8)

                    if self.colormap != 'none':
                        colormap_func = getattr(cv2, f'COLORMAP_{self.colormap.upper()}', cv2.COLORMAP_INFERNO)
                        output_frame = cv2.applyColorMap(depth_uint8, colormap_func)
                    else:
                        output_frame = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

                    writer.write(output_frame)
                    self.progress.emit(i + 1)
            
            if self.is_running:
                self.log.emit(f"✅ Video saved to '{output_path}'")

        finally:
            if cap: cap.release()
            if writer: writer.release()

    def stop(self):
        self.is_running = False

# --- Main GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Depth Estimator")
        self.setGeometry(100, 100, 700, 500) # Adjusted height

        # --- Central Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # --- Folder Selection Widgets ---
        self.layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_path = self._create_folder_selector()
        
        self.layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_path = self._create_folder_selector()

        # --- Options ---
        options_layout = QHBoxLayout()
        self.layout.addLayout(options_layout)

        options_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['none', 'inferno', 'viridis', 'magma', 'jet', 'plasma'])
        options_layout.addWidget(self.colormap_combo)
        
        options_layout.addWidget(QLabel("Max Width:"))
        self.max_res_input = QLineEdit()
        self.max_res_input.setPlaceholderText("e.g., 1024 (optional)")
        options_layout.addWidget(self.max_res_input)
        
        # --- Controls ---
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        # --- Progress and Logging ---
        self.progress_label = QLabel("Waiting to start...")
        self.layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)
        
        self.thread = None
        self.worker = None

    def _create_folder_selector(self):
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
        if dir_path:
            line_edit.setText(dir_path)

    def start_processing(self):
        input_folder = self.input_folder_path.text()
        output_folder = self.output_folder_path.text()
        model_path = "checkpoints/bridge.pth" # Hardcoded model path
        max_res_str = self.max_res_input.text()

        if not all([input_folder, output_folder]):
            self.log_box.append("❌ Please select the input and output folders.")
            return

        if not os.path.exists(model_path):
            self.log_box.append(f"❌ Error: Model not found at the default location: '{model_path}'")
            return

        os.makedirs(output_folder, exist_ok=True)
        self.start_button.setEnabled(False)
        self.start_button.setText("Processing...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_box.append(f"✅ Using device: {device}")

        # --- Setup and run the background thread ---
        self.thread = QThread()
        self.worker = ProcessingWorker(
            input_folder, output_folder, model_path, 
            self.colormap_combo.currentText(), device,
            max_res_str
        )
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.on_processing_finished)
        
        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.video_started.connect(self.on_video_started)

        self.thread.start()

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
        """Handle closing the window while processing."""
        if self.worker:
            self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
