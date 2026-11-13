# xr_teleoperate

Stream a camera feed to a Meta Quest VR headset and capture controller poses as UDP packets.

---

## Overview

| Component | What it does |
|---|---|
| `teleop/teleimager/` | WebRTC camera server — streams video to the Quest browser over HTTPS |
| `udp_receiver.py` | Standalone script that receives and prints controller pose UDP packets |
| `teleop/teleimager/cam_config_server.yaml` | All runtime configuration (camera source, ports, UDP relay destination) |

---

## 1. Prerequisites

- **Conda** with a Python 3.10 environment (called `tv` below)
- **Meta Quest** browser
- The Quest and the PC must be on the **same Wi-Fi network**
- ROS2 (Humble or later) + `cv_bridge` — only needed for the `ros2` camera type

---

## 2. Install

```bash
git clone <this-repo> xr_teleoperate
cd xr_teleoperate

conda activate tv

# Install the camera/WebRTC server
pip install -e teleop/teleimager/
```

---

## 3. SSL Certificate (one-time)

The Quest browser requires HTTPS. Generate a self-signed cert inside the teleimager folder:

```bash
cd teleop/teleimager
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem \
    -subj "/CN=<YOUR_PC_IP>"
```

Replace `<YOUR_PC_IP>` with your machine's LAN IP (e.g. `192.168.0.65`).

The server expects `cert.pem` and `key.pem` to be in `teleop/teleimager/`.

---

## 4. Configure

Edit `teleop/teleimager/cam_config_server.yaml`.

### 4a. Webcam (OpenCV)

```yaml
head_camera:
  enable_webrtc: true
  webrtc_port: 60001
  webrtc_codec: h264
  type: opencv
  image_shape: [480, 640]
  fps: 30
  video_id: 0          # /dev/video0 — change if needed
  serial_number: null
  physical_path: null

udp_relay:
  host: 127.0.0.1      # IP of the machine running udp_receiver.py
  port: 9870
```

### 4b. ROS2 topic (`sensor_msgs/Image`)

```yaml
head_camera:
  enable_webrtc: true
  webrtc_port: 60001
  webrtc_codec: h264
  type: ros2
  ros2_topic: /camera/image   # your topic
  image_shape: [480, 640]     # resize target; match actual resolution to skip resize
  fps: 30

udp_relay:
  host: 127.0.0.1
  port: 9870
```

> The server subscribes to the ROS2 topic on whichever machine it runs on.
> It just needs to be on the same DDS network as the publisher.
> Source your ROS2 workspace before launching.

### Sending controller data to a different host/port

Change `udp_relay.host` and `udp_relay.port` to point at any machine on the network:

```yaml
udp_relay:
  host: 192.168.0.42   # machine that will process controller poses
  port: 9870
```

---

## 5. Run

Open **two terminals**, both with `conda activate tv`.

### Terminal 1 — camera + WebRTC server

```bash
teleimager-server
```


You should see:
```
[Image Server] Image server has started, waiting for client connections...
```

### Terminal 2 — UDP receiver (prints incoming controller poses)

```bash
python udp_receiver.py          # listens on 0.0.0.0:9870 by default
```

To listen on a different port:

```bash
python udp_receiver.py --port 9999
```

---

## 6. Connect from the Quest - do ssh port fowrarding with:  ssh -L 0.0.0.0:60001:localhost:60001 mzucchelli@mrl-122

1. Put on the headset and open the **Quest Browser**.
2. Navigate to `https://<YOUR_PC_IP>:60001/`
3. Accept the self-signed certificate warning.
4. Press **"Start Video"** — you should see the camera feed.
5. Press **"Enter VR & Track Controllers"** — the browser enters immersive VR mode; the video renders on a 3 m wide quad in front of you.
6. Controller poses are sent as UDP packets to the address in `udp_relay` on every XR animation frame.

---

## 7. UDP Packet Format

Packets are little-endian binary, 230 bytes:

| Field | Type | Size |
|---|---|---|
| Magic `0xABCD` | uint16 | 2 B |
| Version `1` | uint8 | 1 B |
| Flags | uint8 | 1 B |
| Head pose (4×4 float32) | float32[16] | 64 B |
| Left controller pose (4×4 float32) | float32[16] | 64 B |
| Right controller pose (4×4 float32) | float32[16] | 64 B |
| Triggers, squeeze, joystick axes (×8) | float32[8] | 32 B |
| Button bitmask | uint16 | 2 B |

Struct format string: `'<2sBB16f16f16f8fH'`

Poses are in the robot coordinate convention (X forward, Y left, Z up), transformed from the WebXR OpenXR frame.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `NET::ERR_CERT_AUTHORITY_INVALID` in browser | Accept the warning / proceed anyway — self-signed cert is expected |
| Black screen in VR | Check that the camera is opening: look for errors in Terminal 1 |
| Wrong camera (`/dev/video1` is a metadata node) | Probe with `python -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(6)]"` and set `video_id` accordingly |
| `ModuleNotFoundError: No module named 'uvc'` | Harmless — `pyuvc` is optional; the server falls back to OpenCV |
| No UDP packets arriving | Make sure `udp_relay.host` matches the IP of the machine running `udp_receiver.py`; check firewall rules for the UDP port |
| ROS2 topic not received | Source your ROS2 workspace before running the server; confirm `ros2 topic echo /camera/image` works |
