import os
import tkinter as tk
from tkinter import messagebox

from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageGrab
from ultralytics import YOLO
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26s.pt"

# Minimum confidence
CONFIDENCE = 0.25

# Window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


# ============================================================
# YOLO MODEL
# ============================================================

print(f"Loading model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# APPLICATION
# ============================================================

class YOLODetector:

    def __init__(self, root):

        self.root = root

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.root.title("YOLO26s Object Detection")

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            800,
            600
        )

        # GNOME/Linux close button
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_window
        )

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.current_image = None
        self.tk_image = None

        # ----------------------------------------------------
        # Main background
        # ----------------------------------------------------

        self.root.configure(
            bg="#f2f2f2"
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = tk.Frame(
            root,
            bg="#f2f2f2"
        )

        header.pack(
            fill="x",
            padx=20,
            pady=(15, 5)
        )

        title = tk.Label(
            header,
            text="YOLO26s Object Detection",
            font=("Sans", 20, "bold"),
            bg="#f2f2f2",
            fg="#222222"
        )

        title.pack()

        subtitle = tk.Label(
            header,
            text=(
                "Drag & drop an image • "
                "Ctrl+V to paste from clipboard"
            ),
            font=("Sans", 11),
            bg="#f2f2f2",
            fg="#666666"
        )

        subtitle.pack(
            pady=(3, 0)
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = tk.Frame(
            root,
            bg="#f2f2f2"
        )

        button_frame.pack(
            pady=10
        )

        self.clear_button = tk.Button(
            button_frame,
            text="Clear Photo",
            font=("Sans", 11),
            padx=18,
            pady=6,
            command=self.clear_photo,
            cursor="hand2"
        )

        self.clear_button.pack(
            side="left",
            padx=5
        )

        self.close_button = tk.Button(
            button_frame,
            text="Close",
            font=("Sans", 11),
            padx=18,
            pady=6,
            command=self.close_window,
            cursor="hand2"
        )

        self.close_button.pack(
            side="left",
            padx=5
        )

        # ----------------------------------------------------
        # Drop area
        # ----------------------------------------------------

        self.drop_area = tk.Frame(
            root,
            bg="#ffffff",
            highlightthickness=2,
            highlightbackground="#cccccc",
            highlightcolor="#3584e4"
        )

        self.drop_area.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(5, 20)
        )

        # Enable drag & drop
        self.drop_area.drop_target_register(
            DND_FILES
        )

        self.drop_area.dnd_bind(
            "<<Drop>>",
            self.drop_image
        )

        # Initial screen
        self.show_drop_message()

        # ----------------------------------------------------
        # Clipboard
        # ----------------------------------------------------

        self.root.bind(
            "<Control-v>",
            self.paste_clipboard
        )

        # Some Linux desktops can send this
        self.root.bind(
            "<Control-V>",
            self.paste_clipboard
        )

    # ========================================================
    # DROP MESSAGE
    # ========================================================

    def show_drop_message(self):

        # Remove existing widgets
        for widget in self.drop_area.winfo_children():
            widget.destroy()

        self.drop_label = tk.Label(
            self.drop_area,
            text=(
                "DROP IMAGE HERE\n\n"
                "or\n\n"
                "CTRL + V"
            ),
            font=("Sans", 20, "bold"),
            bg="#ffffff",
            fg="#666666"
        )

        self.drop_label.pack(
            expand=True
        )

        # Make label accept drag & drop
        self.drop_label.drop_target_register(
            DND_FILES
        )

        self.drop_label.dnd_bind(
            "<<Drop>>",
            self.drop_image
        )

    # ========================================================
    # CLEAR PHOTO
    # ========================================================

    def clear_photo(self):

        self.current_image = None
        self.tk_image = None

        self.show_drop_message()

        print("Photo cleared.")

    # ========================================================
    # CLOSE WINDOW
    # ========================================================

    def close_window(self):

        print("Closing application...")

        self.root.destroy()

    # ========================================================
    # DRAG & DROP
    # ========================================================

    def drop_image(self, event):

        try:

            files = self.root.tk.splitlist(
                event.data
            )

            for file_path in files:

                file_path = os.path.abspath(
                    file_path
                )

                print(
                    f"Received: {file_path}"
                )

                if self.is_image(file_path):

                    self.detect_image(
                        file_path
                    )

                else:

                    messagebox.showwarning(
                        "Invalid File",
                        f"Not an image:\n\n{file_path}"
                    )

        except Exception as e:

            messagebox.showerror(
                "Drag & Drop Error",
                str(e)
            )

    # ========================================================
    # CLIPBOARD
    # ========================================================

    def paste_clipboard(self, event=None):

        try:

            clipboard = ImageGrab.grabclipboard()

            # -----------------------------------------------
            # Clipboard contains an image
            # -----------------------------------------------

            if isinstance(
                clipboard,
                Image.Image
            ):

                print(
                    "Image pasted from clipboard."
                )

                self.detect_pil_image(
                    clipboard
                )

                return

            # -----------------------------------------------
            # Clipboard contains image file(s)
            # -----------------------------------------------

            if isinstance(
                clipboard,
                list
            ):

                for file_path in clipboard:

                    if self.is_image(
                        file_path
                    ):

                        self.detect_image(
                            file_path
                        )

                return

            # -----------------------------------------------
            # Nothing useful in clipboard
            # -----------------------------------------------

            messagebox.showwarning(
                "Clipboard",
                "No image found in clipboard."
            )

        except Exception as e:

            messagebox.showerror(
                "Clipboard Error",
                str(e)
            )

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    def is_image(self, path):

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff"
        )

        return path.lower().endswith(
            extensions
        )

    # ========================================================
    # LOAD IMAGE FROM FILE
    # ========================================================

    def detect_image(self, image_path):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            print(
                f"Running YOLO on: {image_path}"
            )

            self.detect_pil_image(
                image
            )

        except Exception as e:

            messagebox.showerror(
                "Image Error",
                str(e)
            )

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def detect_pil_image(self, image):

        try:

            # PIL -> NumPy
            image_np = np.array(
                image
            )

            # -----------------------------------------------
            # Run YOLO
            # -----------------------------------------------

            results = model.predict(
                source=image_np,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]

            # -----------------------------------------------
            # Draw bounding boxes
            # -----------------------------------------------

            annotated = result.plot()

            # YOLO/OpenCV BGR -> RGB
            annotated = annotated[
                :, :, ::-1
            ]

            annotated_image = Image.fromarray(
                annotated
            )

            # -----------------------------------------------
            # Display
            # -----------------------------------------------

            self.show_image(
                annotated_image
            )

            # -----------------------------------------------
            # Terminal output
            # -----------------------------------------------

            print()
            print("=" * 60)
            print("DETECTIONS")
            print("=" * 60)

            if len(result.boxes) == 0:

                print(
                    "No objects detected."
                )

            else:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    print(
                        f"{class_name:<25}"
                        f"{confidence * 100:6.2f}%"
                    )

            print("=" * 60)

        except Exception as e:

            messagebox.showerror(
                "YOLO Detection Error",
                str(e)
            )

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    def show_image(self, image):

        self.current_image = image

        # Available display area
        max_width = 1100
        max_height = 620

        display_image = image.copy()

        # Keep aspect ratio
        display_image.thumbnail(
            (
                max_width,
                max_height
            ),
            Image.Resampling.LANCZOS
        )

        # PIL -> Tkinter
        self.tk_image = ImageTk.PhotoImage(
            display_image
        )

        # Remove old widgets
        for widget in self.drop_area.winfo_children():
            widget.destroy()

        # Image widget
        image_label = tk.Label(
            self.drop_area,
            image=self.tk_image,
            bg="#ffffff"
        )

        image_label.pack(
            expand=True
        )

        # Keep drag & drop enabled
        image_label.drop_target_register(
            DND_FILES
        )

        image_label.dnd_bind(
            "<<Drop>>",
            self.drop_image
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = TkinterDnD.Tk()

    app = YOLODetector(
        root
    )

    root.mainloop()