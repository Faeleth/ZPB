import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import customtkinter as ctk
import cv2
from PIL import Image
from FaceRecognition import FaceRecognition
from Plot import Plot
from tkinter import filedialog


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.init_window()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after_ids = []

        self.model = FaceRecognition(
            "./results/yolo11x_training_epochs300_128/weights/best.pt"
        )

        self.plot = Plot(100)
        self.text_objects = {}

        self.cap = None
        self.is_running = False

    def init_window(self):
        self.geometry("1000x600")
        self.title("Emotion Analysis App")
        ctk.set_appearance_mode("Dark")

        # --- GRID LAYOUT CONFIGURATION ---
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Col 0: Wideo
        # Col 1: Sidebar
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
            command=self.load_camera,
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

        # self.label_stat_1 = ctk.CTkLabel(self.sidebar_frame, text="Emotion: -")
        # self.label_stat_1.pack(pady=5)

        # self.label_stat_2 = ctk.CTkLabel(self.sidebar_frame, text="Confidence: -")
        # self.label_stat_2.pack(pady=5)

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

        ax.set_xlabel("Quantity")

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

    def load_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.update_frame()

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video/Image", "*.mp4 *.avi *.jpg *.png")]
        )
        if file_path:
            frame = cv2.imread(file_path)
            self.display_image(frame)

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

        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)

        after_id = self.after(10, self.update_frame)
        self.after_ids.append(after_id)

    def display_image(self, frame):
        frame_with_recognition, plot_data = self.model.predict(frame)
        class_counts = self.plot.update(plot_data)
        self.update_plot(class_counts)
        pil_image = Image.fromarray(frame_with_recognition)
        ctk_img = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=(640, 480)
        )
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img  # Garbage Collector protection

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
            if emotion in self.text_objects:
                if count < 10:
                    self.text_objects[emotion].remove()
                    del self.text_objects[emotion]
            if count > 10:
                if emotion in self.text_objects:
                    self.text_objects[emotion].remove()
                self.text_objects[emotion] = self.ax.text(
                    bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{confidence:.4f}",
                    va="center",
                    ha="center",
                    color="black",
                    fontsize=10,
                )
            bar.set_width(count)

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

        self.quit()
