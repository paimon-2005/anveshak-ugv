"""GPS-denied local navigation: VO pose + local map + route + coverage + survivor tags."""
import math, json, time
import cv2, numpy as np

YAW_SIGN = 1  # flip to -1 if the map turns the wrong way vs the real turn

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class QuickVO:
    """FAST/Shi-Tomasi + LK flow + essential matrix."""
    def __init__(self, K):
        self.K, self.prev, self.pts, self.mask = K, None, None, None

    def _detect(self, g):
        if self.mask is None:  # ignore sky: top 35% of frame
            self.mask = np.zeros(g.shape, np.uint8)
            self.mask[int(0.35 * g.shape[0]):] = 255
        return cv2.goodFeaturesToTrack(g, 400, 0.01, 12, mask=self.mask)

    def step(self, g):
        if self.prev is None or self.pts is None or len(self.pts) < 60:
            self.prev, self.pts = g, self._detect(g)
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.pts, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, nxt, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm(self.pts - back, axis=2).ravel()
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        p0, p1 = self.pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
        out = None
        if len(p0) >= 30 and np.median(np.linalg.norm(p1 - p0, axis=1)) > 0.3:
            E, m = cv2.findEssentialMat(p1, p0, self.K, cv2.RANSAC, 0.999, 1.0)
            if E is not None and E.shape == (3, 3):
                n, R, t, _ = cv2.recoverPose(E, p1, p0, self.K, mask=m)
                out = (R, t.ravel(), int(n))
        self.prev = g
        self.pts = p1.reshape(-1, 1, 2).astype(np.float32) if len(p1) >= 60 else self._detect(g)
        return out


class LocalNav:
    def __init__(self, K, route=None, speed_mps=0.25, turn_dps=40.0,
                 cell=0.5, size_m=60.0, cover_r=1.0, tol=1.0, alpha=0.98):
        self.K, self.cell, self.s = K, cell, 4
        self.n = int(size_m / cell)
        self.vis = np.zeros((self.n, self.n), np.uint8)
        self.x = self.y = self.yaw = 0.0          # entry point = (0,0), start heading = +x
        self.trail = [(0.0, 0.0)]
        self.route, self.wp, self.tol = list(route or []), 0, tol
        self.speed, self.turn = speed_mps, math.radians(turn_dps)
        self.cover_r, self.alpha = cover_r, alpha
        self.survivors, self.vo, self.vo_ok = [], QuickVO(K), False
        self._mark()

    # ---------- pose ----------
    def process(self, gray, cmd, dt):
        """cmd: 'forward'|'backward'|'left'|'right'|'stop' (what the rover is doing)."""
        out = self.vo.step(gray)
        self.vo_ok = out is not None and out[2] >= 25
        dyaw = None
        if self.vo_ok:
            R = out[0]
            dyaw = -YAW_SIGN * math.atan2(R[0, 2], R[2, 2])
            if abs(dyaw) > 0.35:
                dyaw = None
        if dyaw is None:  # VO lost -> use commanded turn rate
            dyaw = {"left": self.turn * dt, "right": -self.turn * dt}.get(cmd, 0.0)
        if cmd == "stop":
            dyaw = 0.0
        d = {"forward": self.speed * dt, "backward": -self.speed * dt}.get(cmd, 0.0)  # scale from speed
        mid = self.yaw + dyaw / 2
        self.x += d * math.cos(mid)
        self.y += d * math.sin(mid)
        self.yaw = wrap(self.yaw + dyaw)
        self._mark()
        self._advance()

    def fuse_compass(self, mag_yaw):
        """Call when a magnetometer heading (radians) is available. Complementary filter."""
        self.yaw = wrap(self.yaw + (1 - self.alpha) * wrap(mag_yaw - self.yaw))

    # ---------- mission ----------
    def _advance(self):
        if self.wp < len(self.route):
            tx, ty = self.route[self.wp]
            if math.hypot(tx - self.x, ty - self.y) < self.tol:
                self.wp += 1

    @property
    def done(self):
        return bool(self.route) and self.wp >= len(self.route)

    def mission_error(self):
        """Steering error toward current waypoint, same sign as motor_control (+ = right), -1..1."""
        if self.wp >= len(self.route):
            return None
        tx, ty = self.route[self.wp]
        err_left = wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
        return max(-1.0, min(1.0, -err_left / (math.pi / 2)))

    # ---------- coverage ----------
    def _cell(self, x, y):
        return int(round(x / self.cell)) + self.n // 2, self.n // 2 - int(round(y / self.cell))

    def _mark(self):
        c, r = self._cell(self.x, self.y)
        cv2.circle(self.vis, (c, r), max(1, int(self.cover_r / self.cell)), 1, -1)
        lx, ly = self.trail[-1]
        if math.hypot(self.x - lx, self.y - ly) > 0.2:
            self.trail.append((self.x, self.y))

    def segment_status(self):
        pts, out = [(0.0, 0.0)] + self.route, []
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            L = max(math.hypot(b[0] - a[0], b[1] - a[1]), 0.01)
            k = max(2, int(L / 0.25))
            hit = 0
            for j in range(k + 1):
                x, y = a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k
                c, r = self._cell(x, y)
                hit += int(0 <= r < self.n and 0 <= c < self.n and self.vis[r, c] > 0)
            f = hit / (k + 1)
            out.append({"seg": i + 1, "from": a, "to": b, "covered": round(f, 2),
                        "status": "done" if f >= 0.9 else ("partial" if f > 0.2 else "todo")})
        return out

    # ---------- survivors ----------
    def add_survivors(self, boxes, frame_w):
        """boxes: [(x1, y1, x2, y2, conf), ...] from the YOLO person detector."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        for x1, y1, x2, y2, conf in boxes:
            rng = min(max(1.7 * fy / max(y2 - y1, 1), 0.5), 8.0)       # person ~1.7 m tall
            ang = self.yaw - math.atan2((x1 + x2) / 2 - frame_w / 2, fx)
            sx, sy = self.x + rng * math.cos(ang), self.y + rng * math.sin(ang)
            for s in self.survivors:
                if math.hypot(s["x"] - sx, s["y"] - sy) < 1.5:
                    s["hits"] += 1
                    s["x"], s["y"] = 0.7 * s["x"] + 0.3 * sx, 0.7 * s["y"] + 0.3 * sy
                    s["conf"] = max(s["conf"], float(conf))
                    break
            else:
                self.survivors.append({"id": len(self.survivors) + 1, "x": sx, "y": sy,
                                       "hits": 1, "conf": float(conf), "t": time.time()})

    # ---------- output ----------
    def status(self):
        seg = self.segment_status()
        return {"pose_x": round(self.x, 2), "pose_y": round(self.y, 2),
                "yaw_deg": round(math.degrees(self.yaw), 1), "vo_ok": self.vo_ok,
                "waypoint": f"{self.wp}/{len(self.route)}", "mission_done": self.done,
                "coverage_m2": round(float(self.vis.sum()) * self.cell ** 2, 1),
                "segments": seg,
                "route": [[round(float(x), 3), round(float(y), 3)] for x, y in self.route],
                "trail": [[round(float(x), 3), round(float(y), 3)] for x, y in self.trail[-5000:]],
                "survivors": [s for s in self.survivors if s["hits"] >= 3]}

    def _px(self, x, y):
        W = self.n * self.s
        return int(W / 2 + x / self.cell * self.s), int(W / 2 - y / self.cell * self.s)

    def render(self):
        W = self.n * self.s
        img = np.full((W, W, 3), 30, np.uint8)
        img[cv2.resize(self.vis, (W, W), interpolation=cv2.INTER_NEAREST) > 0] = (0, 110, 0)
        pts = [(0.0, 0.0)] + self.route
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, self._px(*a), self._px(*b), (200, 200, 200), 1)
        for r in self.route:
            cv2.circle(img, self._px(*r), 6, (0, 0, 255), 1)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self._px(*p) for p in self.trail], np.int32)], False, (255, 200, 0), 1)
        for s in self.survivors:
            col = (0, 0, 255) if s["hits"] >= 3 else (0, 165, 255)
            cv2.circle(img, self._px(s["x"], s["y"]), 6, col, -1)
            cv2.putText(img, f"S{s['id']}", self._px(s["x"] + 0.4, s["y"] + 0.4), 0, 0.4, col, 1)
        p = self._px(self.x, self.y)
        tip = self._px(self.x + math.cos(self.yaw), self.y + math.sin(self.yaw))
        cv2.arrowedLine(img, p, tip, (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(img, f"({self.x:.1f},{self.y:.1f}) yaw {math.degrees(self.yaw):.0f}", (8, 18), 0, 0.5, (255, 255, 255), 1)
        return img

    def save_report(self, path="mission_report.json"):
        with open(path, "w") as f:
            json.dump(self.status() | {"trail": self.trail[::5]}, f, indent=2)
