
import json
from datetime import datetime, timezone

from pypylon import pylon, genicam


# Terms used to discover potentially relevant camera features.
KEYWORDS = (
    "calib",
    "intrinsic",
    "extrinsic",
    "distortion",
    "focal",
    "principal",
    "pixelpitch",
    "pixel_size",
    "lens",
    "sensorwidth",
    "sensorheight",
    "resolution",
)


def readable_value(node):
    """Return a readable GenICam node value, if available."""
    try:
        if genicam.IsReadable(node):
            return node.ToString()
    except Exception:
        pass
    return None


def main():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        raise RuntimeError(
            "No camera found. Check the connection, driver, "
            "and manufacturer SDK."
        )

    # Select the first camera. Change this if multiple cameras are connected.
    camera = pylon.InstantCamera(
        factory.CreateDevice(devices[0])
    )

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": {},
        "calibration_related_features": {},
    }

    try:
        camera.Open()

        # Device identity
        for name in (
            "DeviceVendorName",
            "DeviceModelName",
            "DeviceSerialNumber",
            "DeviceFirmwareVersion",
        ):
            value = readable_value(camera.NodeMap.GetNode(name))
            if value is not None:
                results["device"][name] = value

        print("Device:", results["device"])

        # Search all available GenICam features.
        for node in camera.NodeMap.GetNodes():
            try:
                name = node.GetName()
                if not any(
                    term in name.lower() for term in KEYWORDS
                ):
                    continue

                value = readable_value(node)

                results["calibration_related_features"][name] = value

                print(f"{name}: {value}")

            except Exception:
                continue

        with open(
            "camera_metadata.json", "w", encoding="utf-8"
        ) as file:
            json.dump(results, file, indent=2)

        print("\nSaved results to camera_metadata.json")

        if not results["calibration_related_features"]:
            print(
                "No matching feature names found. "
                "This does not prove calibration data is absent; "
                "check the manufacturer SDK and calibration files."
            )

    finally:
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
