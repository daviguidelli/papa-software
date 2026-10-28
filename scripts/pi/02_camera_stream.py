"""Live camera view in any browser on your network.

  python3 scripts/pi/02_camera_stream.py
  then open  http://<pi-ip>:8000  on your laptop (or on the Pi).

Low resolution on purpose: it is for aiming and focusing, not detection.
"""
import io
import socketserver
import sys
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

PAGE = b"""<html><body style="margin:0;background:#111">
<img src="stream.mjpg" style="width:100%;height:100vh;object-fit:contain"></body></html>"""


class Frames(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.cond = Condition()

    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


frames = Frames()


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with frames.cond:
                        frames.cond.wait()
                        data = frames.frame
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


class Server(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (1280, 720)}))
    cam.start_recording(JpegEncoder(), FileOutput(frames))
    print("streaming on port 8000. Ctrl+C to stop.")
    try:
        Server(("", 8000), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop_recording()
    return 0


if __name__ == "__main__":
    sys.exit(main())
