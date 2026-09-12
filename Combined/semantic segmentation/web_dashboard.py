import json
import mimetypes
import threading
import time
import tempfile
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
DASHBOARD_DIRECTORY = Path(__file__).with_name('dashboard')
ALLOWED_COMMANDS = {'forward', 'backward', 'left', 'right', 'stop'}
from motor_control import RoverController

class DashboardState:

    def __init__(self, controller):
        self.controller = controller
        self.condition = threading.Condition()
        self.jpeg_frame = None
        self.frame_sequence = 0
        self.view_frames = {}
        self.telemetry = {'camera_connected': False, 'planner_status': 'WAITING', 'fps': 0.0, 'tracked_features': 0, 'position': [0.0, 0.0, 0.0], 'path_points': 0, 'frame_number': 0}

    def publish(self, frame, telemetry):
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.condition:
            self.telemetry.update(telemetry)
            if success:
                self.jpeg_frame = encoded.tobytes()
                self.frame_sequence += 1
            self.condition.notify_all()

    def update_telemetry(self, telemetry):
        with self.condition:
            self.telemetry.update(telemetry)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            result = dict(self.telemetry)
        result.update(self.controller.status())
        result['timestamp'] = time.time()
        return result

    def publish_views(self, views, telemetry):
        encoded = {}
        for name, frame in views.items():
            ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                encoded[name] = data.tobytes()
        with self.condition:
            self.view_frames = encoded
            self.jpeg_frame = encoded.get('all')
            self.telemetry.update(telemetry)
            self.frame_sequence += 1
            self.condition.notify_all()

    def wait_for_frame(self, after_sequence):
        with self.condition:
            self.condition.wait_for(lambda: self.frame_sequence > after_sequence or not self.controller.running, timeout=2.0)
            return (self.frame_sequence, self.jpeg_frame)

class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = 'AnveshakDashboard/1.0'

    def log_message(self, format_string, *args):
        return

    @property
    def dashboard(self):
        return self.server.dashboard

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length < 0 or content_length > 4096:
            raise ValueError('Request is too large')
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body or b'{}')
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        return payload

    def _serve_static(self, request_path):
        relative_path = 'index.html' if request_path == '/' else request_path.lstrip('/')
        target = (DASHBOARD_DIRECTORY / relative_path).resolve()
        if DASHBOARD_DIRECTORY.resolve() not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _stream_video(self):
        view = parse_qs(urlparse(self.path).query).get('view', ['all'])[0]
        if view not in {'all', 'vo', 'segmentation', 'astar'}:
            self.send_error(400)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        sequence = 0
        try:
            while self.dashboard.controller.running:
                sequence, jpeg_frame = self.dashboard.state.wait_for_frame(sequence)
                with self.dashboard.state.condition:
                    jpeg_frame = self.dashboard.state.view_frames.get(view, jpeg_frame)
                if jpeg_frame is None:
                    continue
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpeg_frame)}\r\n\r\n'.encode('ascii'))
                self.wfile.write(jpeg_frame)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == '/api/status':
            self._send_json(self.dashboard.state.snapshot())
        elif request_path == '/stream.mjpg':
            self._stream_video()
        else:
            self._serve_static(request_path)

    def do_POST(self):
        request_path = urlparse(self.path).path
        try:
            if request_path == '/api/upload':
                self._upload_video()
                return
            payload = self._read_json()
            if request_path == '/api/session/start':
                self.dashboard.open_session()
            elif request_path == '/api/session/heartbeat':
                self.dashboard.heartbeat()
            elif request_path == '/api/source/live':
                self.dashboard.queue_source(None)
            elif request_path == '/api/vo/reset':
                self.dashboard.open_session()
            elif request_path == '/api/mode':
                self.dashboard.controller.set_mode(payload.get('mode'))
            elif request_path == '/api/drive':
                self.dashboard.controller.manual_drive(payload.get('command'), payload.get('speed', 55))
            elif request_path == '/api/emergency':
                self.dashboard.controller.set_emergency(payload.get('active', True))
            else:
                self._send_json({'error': 'Unknown endpoint'}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(self.dashboard.state.snapshot())
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({'error': str(error)}, HTTPStatus.BAD_REQUEST)

    def _upload_video(self):
        length = int(self.headers.get('Content-Length', '0'))
        suffix = Path(self.headers.get('X-Filename', 'test.mp4')).suffix.lower()
        if not 0 < length <= 512 * 1024 * 1024 or suffix not in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}:
            raise ValueError('Upload a supported video of at most 512 MB')
        target = Path(self.dashboard.upload_directory.name) / (uuid.uuid4().hex + suffix)
        self.connection.settimeout(30)
        try:
            with target.open('wb') as output:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1048576, remaining))
                    if not chunk:
                        raise ValueError('Upload interrupted')
                    output.write(chunk)
                    remaining -= len(chunk)
            capture = cv2.VideoCapture(str(target))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                raise ValueError('The uploaded file cannot be decoded as video')
            self.dashboard.queue_source(str(target))
            self._send_json({'uploaded': True, 'message': 'Test video queued; motors disabled'})
        except Exception:
            target.unlink(missing_ok=True)
            raise

class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class DashboardServer:

    def __init__(self, host='0.0.0.0', port=8080, dry_run=False):
        self.session_lock = threading.Lock()
        self.session_revision = 0
        self.session_deadline = 0.0
        self.pending_source = None
        self.upload_directory = tempfile.TemporaryDirectory(prefix='anveshak-video-')
        self.http_server = DashboardHTTPServer((host, int(port)), DashboardRequestHandler)
        try:
            self.controller = RoverController(dry_run=dry_run)
        except Exception:
            self.http_server.server_close()
            raise
        self.state = DashboardState(self.controller)
        self.http_server.dashboard = self
        self.thread = threading.Thread(target=self.http_server.serve_forever, name='dashboard-http-server', daemon=True)
        self.host = host
        self.port = self.http_server.server_address[1]

    def start(self):
        self.thread.start()

    def open_session(self):
        self.controller.stop_motion()
        with self.session_lock:
            self.session_revision += 1
            self.session_deadline = time.monotonic() + 6

    def heartbeat(self):
        with self.session_lock:
            if self.session_revision:
                self.session_deadline = time.monotonic() + 6

    def session(self):
        with self.session_lock:
            return self.session_revision, time.monotonic() < self.session_deadline

    def queue_source(self, path):
        self.controller.set_mode('manual')
        with self.controller.lock:
            self.controller.dry_run = True
        with self.session_lock:
            self.pending_source = (path,)
            self.session_revision += 1
            self.session_deadline = time.monotonic() + 6

    def take_source(self):
        with self.session_lock:
            pending, self.pending_source = self.pending_source, None
            return pending

    def publish(self, frame, telemetry):
        self.state.publish(frame, telemetry)

    def camera_disconnected(self):
        self.controller.stop_motion()
        self.state.update_telemetry({'camera_connected': False, 'planner_status': 'CAMERA OFFLINE'})

    def stop(self):
        self.controller.close()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.thread.join(timeout=1.0)
        self.upload_directory.cleanup()
