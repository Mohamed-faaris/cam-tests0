import asyncio
import os
import threading
import tkinter as tk

import cv2
import numpy as np

from PIL import Image, ImageTk
from ultralytics import YOLO

from dbus_next.aio import MessageBus
from dbus_next import BusType, Message
from dbus_next.constants import MessageType
from dbus_next import Variant


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolo26s.pt"
CONFIDENCE = 0.25

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO26s...")

model = YOLO(MODEL_PATH)

print("YOLO26s loaded.")


# ============================================================
# WAYLAND / XDG PORTAL SCREENSHOT
# ============================================================

async def take_portal_screenshot():

    bus = await MessageBus(
        bus_type=BusType.SESSION
    ).connect()

    # --------------------------------------------------------
    # Unique request token
    # --------------------------------------------------------

    token = "yolo" + os.urandom(8).hex()

    # IMPORTANT:
    # dbus-next requires Variant objects for "v" values.
    options = {
        "handle_token": Variant("s", token),
        "interactive": Variant("b", True),
    }

    # --------------------------------------------------------
    # Call Screenshot portal
    # --------------------------------------------------------

    reply = await bus.call(
        Message(
            destination="org.freedesktop.portal.Desktop",
            path="/org/freedesktop/portal/desktop",
            interface="org.freedesktop.portal.Screenshot",
            member="Screenshot",
            signature="sa{sv}",
            body=[
                "",
                options,
            ],
        )
    )

    if reply.message_type == MessageType.ERROR:

        raise RuntimeError(
            f"Portal error: {reply.body}"
        )

    request_path = reply.body[0]

    # --------------------------------------------------------
    # Wait for Request.Response
    # --------------------------------------------------------

    loop = asyncio.get_running_loop()

    future = loop.create_future()

    def message_handler(message):

        if message.message_type != MessageType.SIGNAL:
            return

        if message.path != request_path:
            return

        if message.interface != "org.freedesktop.portal.Request":
            return

        if message.member != "Response":
            return

        response_code = message.body[0]
        results = message.body[1]

        if future.done():
            return

        if response_code != 0:

            future.set_exception(
                RuntimeError(
                    f"Screenshot cancelled "
                    f"(response={response_code})"
                )
            )

        else:

            future.set_result(
                results
            )

    bus.add_message_handler(
        message_handler
    )

    results = await future

    # --------------------------------------------------------
    # Extract URI
    # --------------------------------------------------------

    uri = results.get("uri")

    if uri is None:

        raise RuntimeError(
            "Portal did not return a screenshot URI."
        )

    if uri.startswith("file://"):

        path = uri[7:]

    else:

        path = uri

    return path


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

        self.root.minsize(
            800,
            600
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
        # Header
        # ----------------------------------------------------

        title = tk.Label(
            root,
            text="YOLO26s Screen Detection",
            font=("Sans", 20, "bold"),
            bg="#f2f2f2",
            fg="#222222",
        )

        title.pack(
            pady=(15, 5)
        )

        subtitle = tk.Label(
            root,
            text="GNOME Wayland • XDG Desktop Portal",
            font=("Sans", 11),
            bg="#f2f2f2",
            fg="#666666",
        )

        subtitle.pack(
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

        self.capture_button = tk.Button(
            button_frame,
            text="Capture Screen",
            font=("Sans", 11),
            padx=20,
            pady=7,
            command=self.capture_screen,
            cursor="hand2",
        )

        self.capture_button.pack(
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
            cursor="hand2",
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
            cursor="hand2",
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
            highlightbackground="#cccccc",
        )

        self.image_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20,
        )

        self.show_empty()

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
    # CAPTURE SCREEN
    # ========================================================

    def capture_screen(self, event=None):

        if str(
            self.capture_button["state"]
        ) == "disabled":

            return

        self.capture_button.config(
            state="disabled"
        )

        print()
        print("Opening Wayland screen capture...")

        thread = threading.Thread(
            target=self.capture_worker,
            daemon=True,
        )

        thread.start()

    # ========================================================
    # BACKGROUND WORKER
    # ========================================================

    def capture_worker(self):

        screenshot_path = None

        try:

            # ------------------------------------------------
            # Run async portal capture
            # ------------------------------------------------

            screenshot_path = asyncio.run(
                take_portal_screenshot()
            )

            print(
                f"Screenshot received: "
                f"{screenshot_path}"
            )

            # ------------------------------------------------
            # Read screenshot
            # ------------------------------------------------

            image = cv2.imread(
                screenshot_path
            )

            if image is None:

                raise RuntimeError(
                    "Could not read screenshot."
                )

            print(
                f"Screen size: "
                f"{image.shape[1]}x"
                f"{image.shape[0]}"
            )

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            print(
                "Running YOLO26s..."
            )

            results = model.predict(
                source=image,
                conf=CONFIDENCE,
                verbose=False,
            )

            result = results[0]

            # ------------------------------------------------
            # Draw detections
            # ------------------------------------------------

            annotated = result.plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            output = Image.fromarray(
                annotated
            )

            # ------------------------------------------------
            # Send image back to Tkinter thread
            # ------------------------------------------------

            self.root.after(
                0,
                lambda img=output:
                self.show_image(img)
            )

            # ------------------------------------------------
            # Terminal detections
            # ------------------------------------------------

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

            print()
            print(
                f"Screen capture error: {e}"
            )

            self.root.after(
                0,
                lambda error=str(e):
                self.show_error(error)
            )

        finally:

            # ------------------------------------------------
            # Remove temporary screenshot
            # ------------------------------------------------

            if screenshot_path:

                try:

                    os.remove(
                        screenshot_path
                    )

                except OSError:

                    pass

            self.root.after(
                0,
                lambda:
                self.capture_button.config(
                    state="normal"
                )
            )

    # ========================================================
    # SHOW IMAGE
    # ========================================================

    def show_image(self, image):

        for widget in self.image_frame.winfo_children():

            widget.destroy()

        display_image = image.copy()

        display_image.thumbnail(
            (
                WINDOW_WIDTH - 80,
                WINDOW_HEIGHT - 180,
            ),
            Image.Resampling.LANCZOS,
        )

        self.tk_image = ImageTk.PhotoImage(
            display_image
        )

        image_label = tk.Label(
            self.image_frame,
            image=self.tk_image,
            bg="white",
        )

        image_label.pack(
            expand=True
        )

    # ========================================================
    # ERROR
    # ========================================================

    def show_error(self, error):

        for widget in self.image_frame.winfo_children():

            widget.destroy()

        label = tk.Label(
            self.image_frame,
            text=(
                "SCREEN CAPTURE ERROR\n\n"
                + error
            ),
            font=("Sans", 14),
            bg="white",
            fg="#cc0000",
            wraplength=900,
        )

        label.pack(
            expand=True
        )

    # ========================================================
    # EMPTY
    # ========================================================

    def show_empty(self):

        for widget in self.image_frame.winfo_children():

            widget.destroy()

        label = tk.Label(
            self.image_frame,
            text=(
                "SCREEN CAPTURE\n\n"
                "Press SPACE\n"
                "or click Capture Screen"
            ),
            font=("Sans", 20, "bold"),
            bg="white",
            fg="#666666",
        )

        label.pack(
            expand=True
        )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        self.tk_image = None

        self.show_empty()

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