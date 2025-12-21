import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import customtkinter as ctk
import cv2
from PIL import Image
from FaceRecognition import FaceRecognition
from Plot import Plot
from tkinter import filedialog, messagebox
import os


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.FRAMES_TO_REMEMBER = 100

        self.init_window()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after_ids = []

        self.model = FaceRecognition(
            "./results/yolo11x_training_epochs300_128/weights/best.pt"
        )

        self.plot = Plot(self.FRAMES_TO_REMEMBER)
        self.text_objects = {}

        self.cap = None
        self.is_running = False

        # Keep track of the temp file to clean it up later
        self.temp_video_path = "temp_downscaled.mp4"

    def reset_plot(self):
        cc = {
            "Anger": [0,0.0],
            "Contempt": [0,0.0],
            "Disgust": [0,0.0],
            "Fear": [0,0.0],
            "Happy": [0,0.0],
            "Neutral": [0,0.0],
            "Sad": [0,0.0],
            "Surprise": [0,0.0],
        }
        self.update_plot(cc)
        self.plot = Plot(self.FRAMES_TO_REMEMBER)
        self.text_objects = {}
        

    def init_window(self):
        self.geometry("1000x600")
        self.title("Emotion Analysis App")
        ctk.set_appearance_mode("Dark")

        # --- GRID LAYOUT CONFIGURATION ---
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

        # ============================================================
        # TOP FRAME
        # ============================================================
        self.top_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.top_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )

        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)

        self.btn_camera = ctk.CTkButton(
            self.top_frame,
            text="Load Camera",
            command=self.camera_load_button,
            height=40,
        )

        self.btn_camera.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_load = ctk.CTkButton(
            self.top_frame,
            text="Load Video/Image",
            command=self.load_file,
            height=40,
        )
        self.btn_load.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # ============================================================
        # MAIN VIDEO AREA
        # ============================================================
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(
            row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10)
        )

        self.video_label = ctk.CTkLabel(
            self.video_frame, text="Load Camera/Video/Image"
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # ============================================================
        # SIDEBAR
        # ============================================================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(
            row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10)
        )

        self.sidebar_label = ctk.CTkLabel(
            self.sidebar_frame, text="Statistics / Plots", font=("Arial", 16, "bold")
        )
        self.sidebar_label.pack(pady=20)

        self.plot_placeholder = ctk.CTkFrame(
            self.sidebar_frame, height=150, fg_color="#2b2b2b"
        )
        self.plot_placeholder.pack(fill="x", padx=10, pady=10)

        self.btn_start_stop = ctk.CTkButton(
            self.sidebar_frame,
            text="Stop",
            command=self.start_stop,
            height=40,
        )
        self.btn_start_stop.pack(side="bottom", fill="x", padx=10, pady=20)

        self.create_bar_chart()

        self.bind("<Configure>", self.resize_plot)

    # ============================================================
    # PLOT
    # ============================================================
    def create_bar_chart(self):
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.subplots_adjust(left=0.24)

        self.emotions = [
            "Anger",
            "Contempt",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]
        values = np.zeros(len(self.emotions))

        self.bars = ax.barh(self.emotions, values, color="#2196F3")
        ax.set_xlim(0, 20)
        ax.set_xlabel("Quantity", color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        fig.patch.set_facecolor("#2b2b2b")
        ax.spines["top"].set_color("#2b2b2b")
        ax.spines["right"].set_color("#2b2b2b")
        ax.spines["left"].set_color("#2196F3")
        ax.spines["bottom"].set_color("#2196F3")
        ax.set_facecolor("#2b2b2b")
        ax.grid(False)

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_placeholder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self.fig = fig
        self.ax = ax

    def resize_plot(self, event):
        window_height = self.sidebar_frame.winfo_height()
        plot_height = window_height - 156
        self.plot_placeholder.configure(height=plot_height)
        left_value = 0.15 + ((1 / self.top_frame.winfo_width()) * 100)
        left_value = max(0, min(left_value, 0.85))
        self.fig.subplots_adjust(left=left_value)

    def camera_load_button(self):
        self.reset_plot()
        self.load_camera()

    def load_file(self):
        self.reset_plot()
        file_path = filedialog.askopenfilename(
            filetypes=[("Video/Image", "*.mp4 *.avi *.jpg *.png")]
        )
        if not file_path:
            return

        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_start_stop.configure(text="Start")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".mp4", ".avi", ".mov"]:
            self.video_label.configure(text="Preprocessing video... please wait...")
            self.update()

            # Downscale the whole video first
            optimized_path = self.preprocess_video(file_path)

            self.cap = cv2.VideoCapture(optimized_path)
            self.is_running = True
            self.is_video_file = True  # Enable skip logic
            self.btn_start_stop.configure(text="Stop")
            self.camera_loop()

        elif ext in [".jpg", ".png", ".jpeg"]:
            frame = cv2.imread(file_path)
            if frame is not None:
                self.display_image(frame)

    def load_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            # camera does not need manual skipping
            self.is_video_file = False
            self.update_frame()

    def preprocess_video(self, input_path):
        """
        Reads the input video. If it's larger than 640px wide, it downscales
        the ENTIRE video to a temporary file before playback starts.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return input_path

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_width = 640

        # If video is already small enough, just return original path
        if width <= target_width:
            cap.release()
            return input_path

        # Calculate new dimensions
        scale_ratio = target_width / width
        new_width = target_width
        new_height = int(height * scale_ratio)

        # Setup Video Writer
        # 'mp4v' is generally safe for mp4 containers in OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.temp_video_path, fourcc, fps, (new_width, new_height)
        )

        print(f"Downscaling video from {width}x{height} to {new_width}x{new_height}...")

        # Process every frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            out.write(resized_frame)

        cap.release()
        out.release()
        print("Downscaling complete.")

        return self.temp_video_path

    def start_stop(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.configure(image=None)
            self.btn_start_stop.configure(text="Start")
        else:
            self.load_camera()
            self.btn_start_stop.configure(text="Stop")

    def camera_loop(self):
        if not self.is_running or not self.cap or not self.winfo_exists():
            return

        # If we are playing a video file, we manually skip frames to simulate real-time speed.
        if getattr(self, "is_video_file", False):
            skip_rate = 2
            for _ in range(skip_rate):
                self.cap.grab()

        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)

            after_id = self.after(10, self.update_frame)
            self.after_ids.append(after_id)
        else:
            self.start_stop()

    def display_image(self, frame):
        frame_with_recognition, plot_data = self.model.predict(frame)

        if not hasattr(self, "frame_count"):
            self.frame_count = 0
        self.frame_count += 1

        class_counts = self.plot.update(plot_data)
        self.update_plot(class_counts)

        # Convert for tkinter
        pil_image = Image.fromarray(frame_with_recognition)

        # Ensure it fits the UI box
        ctk_img = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=(640, 480)
        )
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def update_frame(self):
        if not self.is_running or not self.winfo_exists():
            return
        self.camera_loop()

    def update_plot(self, class_counts):
        new_values = [item[0] for item in class_counts.values()]
        new_confidences = [item[1] for item in class_counts.values()]

        for bar, emotion, count, confidence in zip(
            self.bars, self.emotions, new_values, new_confidences
        ):
            bar.set_width(count)
            if emotion in self.text_objects:
                if count <= 0:
                    self.text_objects[emotion].remove()
                    del self.text_objects[emotion]
            if bar.get_width() > 10:
                if emotion in self.text_objects:
                    self.text_objects[emotion].remove()
                self.text_objects[emotion] = self.ax.text(
                    bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{confidence:.4f}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=10,
                )
            elif bar.get_width() > 0:
                if emotion in self.text_objects:
                    self.text_objects[emotion].remove()
                self.text_objects[emotion] = self.ax.text(
                    bar.get_width() + 5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{confidence:.4f}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=10,
                )
        self.ax.set_xlim(0, max(20, max(new_values) + 10))
        self.canvas.draw()

    def on_close(self):
        self.is_running = False
        for aid in self.after_ids:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
        self.after_ids.clear()
        if self.cap:
            self.cap.release()

        # Clean up temp file
        if os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
            except:
                pass

        self.quit()
