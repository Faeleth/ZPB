import customtkinter as ctk
import cv2
from PIL import Image
from FaceRecognition import FaceRecognition
from tkinter import filedialog


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.init_window()

        self.model = FaceRecognition(
            "./results/yolo11x_training_epochs300_128/weights/best.pt"
        )

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

        self.label_stat_1 = ctk.CTkLabel(self.sidebar_frame, text="Emotion: -")
        self.label_stat_1.pack(pady=5)

        self.label_stat_2 = ctk.CTkLabel(self.sidebar_frame, text="Confidence: -")
        self.label_stat_2.pack(pady=5)

        self.btn_start_stop = ctk.CTkButton(
            self.sidebar_frame,
            text="Stop",
            command=self.start_stop,
            height=40,
        )
        self.btn_start_stop.pack(side="bottom", fill="x", padx=10, pady=20)

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
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.display_image(frame)

            self.after(10, self.update_frame)  # 10 ms

    def display_image(self, frame):
        frame_with_recognition = self.model.predict(frame)
        pil_image = Image.fromarray(frame_with_recognition)
        ctk_img = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=(640, 480)
        )
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img  # Garbage Collector protection

    def update_frame(self):
        self.camera_loop()


if __name__ == "__main__":
    app = App()
    app.mainloop()
