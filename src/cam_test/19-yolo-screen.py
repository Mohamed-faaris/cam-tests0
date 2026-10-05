import tkinter as tk
import numpy as np
import cv2

from PIL import Image, ImageTk, ImageGrab
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolo26s.pt"
CONFIDENCE = 0.25

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO26s...")

model = YOLO(MODEL_PATH)

print("YOLO26s loaded.")


# ============================================================
# APPLICATION
# ============================================================

class ScreenYOLO:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "YOLO26s Screen Detection"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.configure(
            bg="#f2f2f2"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.tk_image = None

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = tk.Label(
            root,
            text="YOLO26s Screen Detection",
            font=("Sans", 20, "bold"),
            bg="#f2f2f2",
            fg="#222222"
        )

        title.pack(
            pady=(15, 5)
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        instructions = tk.Label(
            root,
            text=(
                "Press SPACE to capture screen and detect objects"
                "  •  ESC / Q to close"
            ),
            font=("Sans", 11),
            bg="#f2f2f2",
            fg="#666666"
        )

        instructions.pack(
            pady=(0, 10)
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = tk.Frame(
            root,
            bg="#f2f2f2"
        )

        button_frame.pack(
            pady=5
        )

        capture_button = tk.Button(
            button_frame,
            text="Capture Screen",
            font=("Sans", 11),
            padx=20,
            pady=7,
            command=self.capture_screen,
            cursor="hand2"
        )

        capture_button.pack(
            side="left",
            padx=5
        )

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Sans", 11),
            padx=20,
            pady=7,
            command=self.clear,
            cursor="hand2"
        )

        clear_button.pack(
            side="left",
            padx=5
        )

        close_button = tk.Button(
            button_frame,
            text="Close",
            font=("Sans", 11),
            padx=20,
            pady=7,
            command=self.close,
            cursor="hand2"
        )

        close_button.pack(
            side="left",
            padx=5
        )

        # ----------------------------------------------------
        # Image area
        # ----------------------------------------------------

        self.image_frame = tk.Frame(
            root,
            bg="white",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )

        self.image_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        self.message = tk.Label(
            self.image_frame,
            text=(
                "SCREEN CAPTURE\n\n"
                "Press SPACE\n"
                "or click Capture Screen"
            ),
            font=("Sans", 20, "bold"),
            bg="white",
            fg="#666666"
        )

        self.message.pack(
            expand=True
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        self.root.bind(
            "<space>",
            self.capture_screen
        )

        self.root.bind(
            "<Escape>",
            self.close
        )

        self.root.bind(
            "<q>",
            self.close
        )

        self.root.bind(
            "<Q>",
            self.close
        )

    # ========================================================
    # SCREEN CAPTURE
    # ========================================================

    def capture_screen(self, event=None):

        print("\nCapturing screen...")

        try:

            # -----------------------------------------------
            # Capture entire screen
            # -----------------------------------------------

            screenshot = ImageGrab.grab()

            # PIL -> NumPy
            image = np.array(
                screenshot
            )

            # RGB -> BGR
            image_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            print(
                f"Screen size: "
                f"{image_bgr.shape[1]}x"
                f"{image_bgr.shape[0]}"
            )

            # -----------------------------------------------
            # YOLO
            # -----------------------------------------------

            results = model.predict(
                source=image_bgr,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]

            # -----------------------------------------------
            # Draw detections
            # -----------------------------------------------

            annotated = result.plot()

            # BGR -> RGB
            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            output = Image.fromarray(
                annotated
            )

            # -----------------------------------------------
            # Display
            # -----------------------------------------------

            self.show_image(
                output
            )

            # -----------------------------------------------
            # Print results
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

            print(
                f"Screen capture error: {e}"
            )

    # ========================================================
    # SHOW IMAGE
    # ========================================================

    def show_image(self, image):

        # Remove previous widgets
        for widget in self.image_frame.winfo_children():

            widget.destroy()

        # Make a copy
        display_image = image.copy()

        # Fit inside window
        display_image.thumbnail(
            (
                WINDOW_WIDTH - 80,
                WINDOW_HEIGHT - 180
            ),
            Image.Resampling.LANCZOS
        )

        # PIL -> Tkinter
        self.tk_image = ImageTk.PhotoImage(
            display_image
        )

        image_label = tk.Label(
            self.image_frame,
            image=self.tk_image,
            bg="white"
        )

        image_label.pack(
            expand=True
        )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        self.tk_image = None

        for widget in self.image_frame.winfo_children():

            widget.destroy()

        self.message = tk.Label(
            self.image_frame,
            text=(
                "SCREEN CAPTURE\n\n"
                "Press SPACE\n"
                "or click Capture Screen"
            ),
            font=("Sans", 20, "bold"),
            bg="white",
            fg="#666666"
        )

        self.message.pack(
            expand=True
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self, event=None):

        print("Closing...")

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ScreenYOLO(
        root
    )

    root.mainloop()