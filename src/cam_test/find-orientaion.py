import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = (8, 5)

# Your checkerboard squares are 4 mm
SQUARE_SIZE_MM = 4.0


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

data = np.load("camera_calibration.npz")

camera_matrix = data["camera_matrix"]
distortion = data["distortion"]


print("Camera matrix:")
print(camera_matrix)

print("\nDistortion:")
print(distortion)


# ============================================================
# CREATE CHECKERBOARD 3D POINTS
# ============================================================

object_points = np.zeros(
    (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
    dtype=np.float32
)

object_points[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)

object_points *= SQUARE_SIZE_MM


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("\nPress ESC to exit")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Find checkerboard
    # --------------------------------------------------------

    found, corners = cv2.findChessboardCornersSB(
        gray,
        BOARD_SIZE
    )

    if found:

        # Make sure shape is correct
        image_points = corners.reshape(-1, 2)

        # ----------------------------------------------------
        # Estimate checkerboard pose
        # ----------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            # ------------------------------------------------
            # Convert rotation vector -> rotation matrix
            # ------------------------------------------------

            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # ------------------------------------------------
            # Convert rotation matrix -> Euler angles
            # ------------------------------------------------

            sy = np.sqrt(
                rotation_matrix[0, 0] ** 2 +
                rotation_matrix[1, 0] ** 2
            )

            singular = sy < 1e-6

            if not singular:

                roll = np.arctan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )

                pitch = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                yaw = np.arctan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )

            else:

                roll = np.arctan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )

                pitch = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                yaw = 0

            # Convert radians -> degrees

            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)

            # ------------------------------------------------
            # Translation
            # ------------------------------------------------

            x = tvec[0, 0]
            y = tvec[1, 0]
            z = tvec[2, 0]

            # ------------------------------------------------
            # Draw coordinate axes
            # ------------------------------------------------

            axis_length = 40.0  # mm

            axis_points = np.float32([
                [0, 0, 0],                    # origin
                [axis_length, 0, 0],          # X
                [0, axis_length, 0],          # Y
                [0, 0, -axis_length],         # Z
            ])

            projected, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                camera_matrix,
                distortion
            )

            projected = projected.reshape(-1, 2)

            origin = tuple(
                projected[0].astype(int)
            )

            x_axis = tuple(
                projected[1].astype(int)
            )

            y_axis = tuple(
                projected[2].astype(int)
            )

            z_axis = tuple(
                projected[3].astype(int)
            )

            # X axis
            cv2.line(
                frame,
                origin,
                x_axis,
                (0, 0, 255),
                3
            )

            # Y axis
            cv2.line(
                frame,
                origin,
                y_axis,
                (0, 255, 0),
                3
            )

            # Z axis
            cv2.line(
                frame,
                origin,
                z_axis,
                (255, 0, 0),
                3
            )

            # ------------------------------------------------
            # Draw checkerboard
            # ------------------------------------------------

            cv2.drawChessboardCorners(
                frame,
                BOARD_SIZE,
                corners,
                found
            )

            # ------------------------------------------------
            # Display position
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"X: {x:.1f} mm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y: {y:.1f} mm",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Z: {z:.1f} mm",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Display orientation
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Roll:  {roll_deg:.2f} deg",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch_deg:.2f} deg",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yaw:   {yaw_deg:.2f} deg",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Print periodically
            # ------------------------------------------------

            print(
                f"\r"
                f"Position: "
                f"X={x:8.2f} "
                f"Y={y:8.2f} "
                f"Z={z:8.2f} mm | "
                f"Rotation: "
                f"R={roll_deg:7.2f} "
                f"P={pitch_deg:7.2f} "
                f"Y={yaw_deg:7.2f}",
                end=""
            )

    else:

        cv2.putText(
            frame,
            "CHECKERBOARD NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Checkerboard Pose",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\n\nStopped.")