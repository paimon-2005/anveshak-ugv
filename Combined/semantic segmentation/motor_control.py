import os
import threading
import time
from urllib.parse import quote
from urllib.request import urlopen


ALLOWED_COMMANDS = {"forward", "backward", "left", "right", "stop"}


class RoverController:
    def __init__(self, dry_run=False, timeout=0.45):
        self.command_url_template = os.environ.get("ROVER_COMMAND_URL_TEMPLATE",
            "http://192.168.4.1/action?go={command}&speed={speed}")
        if "{command}" not in self.command_url_template:
            raise ValueError("ROVER_COMMAND_URL_TEMPLATE must contain {command}")
        self.command_values = {c: os.environ.get(f"ROVER_{c.upper()}_VALUE", c) for c in ALLOWED_COMMANDS}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.mode, self.command, self.speed = "manual", "stop", 55
        self.emergency = False
        self.deadline = 0.0
        self.last_sent_at = self.last_error = None
        self.dry_run, self.timeout = bool(dry_run), float(timeout)
        self.revision, self.sent_revision = 0, -1
        self.running = True
        self.worker = threading.Thread(target=self._command_worker, name="rover-command-worker", daemon=True)
        self.worker.start()

    def _change(self, command, speed=None):
        if self.emergency and command != "stop":
            command = "stop"
        speed = self.speed if speed is None else max(0, min(100, int(speed)))
        if speed == 0:
            command = "stop"
        if command != self.command or speed != self.speed:
            self.command, self.speed = command, speed
            self.revision += 1
        self.condition.notify_all()

    def _send_command(self, command, speed):
        if self.dry_run:
            return
        url = self.command_url_template.format(command=quote(self.command_values[command], safe=""), speed=speed)
        with urlopen(url, timeout=self.timeout) as response:
            response.read(64)

    def _command_worker(self):
        next_send = 0.0
        while True:
            with self.condition:
                if not self.running:
                    return
                now = time.monotonic()
                if self.command != "stop" and now >= self.deadline:
                    self._change("stop")
                changed = self.sent_revision != self.revision
                heartbeat = self.command != "stop" or self.last_error is not None
                if not changed and (not heartbeat or now < next_send):
                    self.condition.wait(timeout=0.04)
                    continue
                command, speed, revision = self.command, self.speed, self.revision
            error = None
            try:
                self._send_command(command, speed)
            except Exception as exception:
                error = str(exception)
            with self.condition:
                self.last_error = error
                if error is None:
                    self.last_sent_at, self.sent_revision = time.time(), revision
                elif revision == self.revision and self.command != "stop":
                    self.mode = "manual"
                    self._change("stop")
                if error is not None and command == "stop":
                    self.sent_revision = revision
                next_send = time.monotonic() + 0.2
                self.condition.notify_all()

    def set_mode(self, mode):
        if mode not in {"manual", "auto"}:
            raise ValueError("Mode must be manual or auto")
        with self.condition:
            if self.emergency and mode == "auto":
                raise ValueError("Release the emergency lock first")
            self.mode, self.deadline = mode, 0.0
            self._change("stop")

    def manual_drive(self, command, speed):
        if command not in ALLOWED_COMMANDS:
            raise ValueError("Unsupported drive command")
        speed = int(speed)
        with self.condition:
            self.mode, self.deadline = "manual", time.monotonic() + 0.65
            self._change(command, speed)

    def stop_motion(self):
        with self.condition:
            self.deadline = 0.0
            self._change("stop")

    def set_emergency(self, active):
        if not isinstance(active, bool):
            raise ValueError("active must be a boolean")
        with self.condition:
            self.emergency, self.mode, self.deadline = active, "manual", 0.0
            self._change("stop")

    def update_autonomous(self, planned_path, frame_width, perception_at=None):
        now = time.monotonic()
        with self.condition:
            if self.mode != "auto" or self.emergency:
                return
            fresh = perception_at is not None and 0 <= now - perception_at <= 0.8
            if len(planned_path) < 2 or not fresh or frame_width <= 0:
                self._change("stop")
                return
            target = planned_path[min(5, len(planned_path) - 1)][0]
            error = (target - frame_width / 2) / (frame_width / 2)
            command = "left" if error < -0.16 else "right" if error > 0.16 else "forward"
            self.deadline = min(now + 0.65, perception_at + 0.8)
            self._change(command)

    def status(self):
        with self.lock:
            return {"mode": self.mode, "command": self.command, "speed": self.speed,
                    "emergency": self.emergency, "dry_run": self.dry_run,
                    "controller_connected": not self.dry_run and self.last_sent_at is not None and self.last_error is None,
                    "controller_error": self.last_error, "last_command_at": self.last_sent_at}

    def close(self):
        with self.condition:
            self.emergency, self.mode = True, "manual"
            self._change("stop")
            self.revision += 1
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.sent_revision == self.revision, timeout=self.timeout * 2 + 0.3)
            self.running = False
            self.condition.notify_all()
        self.worker.join(timeout=self.timeout + 0.3)
