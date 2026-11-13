import json
import os
import socket
import threading


class XRUnixReceiver:
    """Unix-socket (AF_UNIX SOCK_DGRAM) receiver for JSON XR headset packets.

    Alternative to XRReceiver when sender and receiver share a host or mount.
    """

    def __init__(self, socket_path: str = "/tmp/xr_pose.sock"):
        self.socket_path_ = socket_path

        self.payload_ = None
        self.payload_lock_ = threading.Lock()
        self.running_ = False
        self.thread_ = None

    def start(self):
        if self.running_:
            return
        self.running_ = True
        self.thread_ = threading.Thread(target=self._receiver_loop, daemon=True)
        self.thread_.start()

    def stop(self):
        self.running_ = False
        if self.thread_:
            self.thread_.join()

    def _receiver_loop(self):
        print("XRUnixReceiver loop started")
        # Remove stale socket file from a previous run
        if os.path.exists(self.socket_path_):
            os.unlink(self.socket_path_)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock_dir = os.path.dirname(self.socket_path_)
            if sock_dir and not os.path.isdir(sock_dir):
                raise FileNotFoundError(
                    f"Socket directory does not exist: '{sock_dir}'. "
                    "Ensure the directory is created and mounted into the container."
                )
            sock.bind(self.socket_path_)
            os.chmod(self.socket_path_, 0o777)
            sock.settimeout(1.0)
            print(f"Listening for XR tracking data on unix:{self.socket_path_}...")

            while self.running_:
                try:
                    data = sock.recv(8192)
                except socket.timeout:
                    continue
                payload = self._decode(data)
                if payload is not None:
                    with self.payload_lock_:
                        self.payload_ = payload
        except Exception as e:
            print(f"[XRUnixReceiver] fatal error in receiver loop: {e}")
            self.running_ = False
        finally:
            sock.close()
            if os.path.exists(self.socket_path_):
                os.unlink(self.socket_path_)

    def _decode(self, data: bytes):
        try:
            return json.loads(data.decode())
        except Exception as e:
            print(f"[XRUnixReceiver] decode error: {e}")
            return None

    def get_payload(self):
        with self.payload_lock_:
            return self.payload_
