const state = {
  mode: "manual",
  emergency: false,
  activeCommand: null,
  heartbeat: null,
  speed: 55,
};

const elements = {
  cameraStatus: document.querySelector("#cameraStatus"),
  controllerStatus: document.querySelector("#controllerStatus"),
  missionClock: document.querySelector("#missionClock"),
  modeReadout: document.querySelector("#modeReadout"),
  plannerReadout: document.querySelector("#plannerReadout"),
  commandReadout: document.querySelector("#commandReadout"),
  pathReadout: document.querySelector("#pathReadout"),
  fpsReadout: document.querySelector("#fpsReadout"),
  authorityBadge: document.querySelector("#authorityBadge"),
  controlNote: document.querySelector("#controlNote"),
  speedRange: document.querySelector("#speedRange"),
  speedValue: document.querySelector("#speedValue"),
  emergencyButton: document.querySelector("#emergencyButton"),
  positionX: document.querySelector("#positionX"),
  positionY: document.querySelector("#positionY"),
  positionZ: document.querySelector("#positionZ"),
  featuresReadout: document.querySelector("#featuresReadout"),
  toast: document.querySelector("#toast"),
  voStatus: document.querySelector("#voStatus"),
  inliersReadout: document.querySelector("#inliersReadout"),
  perceptionAge: document.querySelector("#perceptionAge"),
  floorOverride: document.querySelector("#floorOverride"),
};

const modeButtons = [...document.querySelectorAll("[data-mode]")];
const driveButtons = [...document.querySelectorAll("[data-command]")];
const keyCommands = {
  KeyW: "forward",
  ArrowUp: "forward",
  KeyS: "backward",
  ArrowDown: "backward",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
  Space: "stop",
};

async function api(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Command failed");
  return body;
}

let toastTimer;
function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2400);
}

function setStatusPill(element, online, text) {
  element.classList.toggle("online", online);
  element.querySelector("span:last-child").textContent = text;
}

function applyMode(mode) {
  state.mode = mode;
  modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  driveButtons.forEach((button) => { button.disabled = mode === "auto" || state.emergency; });
  elements.authorityBadge.textContent = mode === "auto" ? "A* AUTO" : "MANUAL";
  elements.modeReadout.textContent = mode === "auto" ? "AUTONOMOUS" : "MANUAL OVERRIDE";
  elements.controlNote.textContent = mode === "auto"
    ? "A* has drive authority. Any directional key immediately returns control to manual."
    : "Manual commands override A* drive output. Release a control to stop.";
}

async function selectMode(mode) {
  stopHeartbeat();
  try {
    const status = await api("/api/mode", { mode });
    applyMode(status.mode);
  } catch (error) {
    showToast(error.message);
  }
}

function sendDrive(command) {
  return api("/api/drive", { command, speed: state.speed });
}

function engageDrive(command) {
  if (state.emergency || state.activeCommand === command) return;
  stopHeartbeat(false);
  state.activeCommand = command;
  applyMode("manual");
  driveButtons.forEach((button) => button.classList.toggle("engaged", button.dataset.command === command));
  sendDrive(command).catch((error) => showToast(error.message));

  if (command !== "stop") {
    state.heartbeat = setInterval(() => {
      sendDrive(command).catch(() => stopHeartbeat());
    }, 250);
  }
}

function stopHeartbeat(sendStop = true) {
  clearInterval(state.heartbeat);
  state.heartbeat = null;
  const wasMoving = state.activeCommand && state.activeCommand !== "stop";
  state.activeCommand = null;
  driveButtons.forEach((button) => button.classList.remove("engaged"));
  if (sendStop && wasMoving) sendDrive("stop").catch(() => {});
}

modeButtons.forEach((button) => button.addEventListener("click", () => selectMode(button.dataset.mode)));

driveButtons.forEach((button) => {
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    button.setPointerCapture?.(event.pointerId);
    engageDrive(button.dataset.command);
  });
  button.addEventListener("pointerup", () => stopHeartbeat());
  button.addEventListener("pointercancel", () => stopHeartbeat());
  button.addEventListener("lostpointercapture", () => stopHeartbeat());
});

window.addEventListener("keydown", (event) => {
  const command = keyCommands[event.code];
  if (!command || event.repeat || state.emergency) return;
  event.preventDefault();
  engageDrive(command);
});

window.addEventListener("keyup", (event) => {
  if (!keyCommands[event.code]) return;
  event.preventDefault();
  stopHeartbeat();
});

window.addEventListener("blur", () => stopHeartbeat());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopHeartbeat();
});

elements.speedRange.addEventListener("input", () => {
  state.speed = Number(elements.speedRange.value);
  elements.speedValue.value = `${state.speed}%`;
  elements.speedValue.textContent = `${state.speed}%`;
});

elements.emergencyButton.addEventListener("click", async () => {
  stopHeartbeat(false);
  const active = !state.emergency;
  try {
    const status = await api("/api/emergency", { active });
    state.emergency = status.emergency;
    elements.emergencyButton.classList.toggle("active", state.emergency);
    elements.emergencyButton.querySelector("strong").textContent = state.emergency ? "Release emergency lock" : "Emergency stop";
    applyMode(status.mode);
  } catch (error) {
    showToast(error.message);
  }
});

function renderStatus(status) {
  state.emergency = Boolean(status.emergency);
  state.speed = Number(status.speed ?? state.speed);
  elements.speedRange.value = state.speed;
  elements.speedValue.value = `${state.speed}%`;
  elements.speedValue.textContent = `${state.speed}%`;
  applyMode(status.mode || "manual");

  setStatusPill(elements.cameraStatus, status.camera_connected, status.camera_connected ? "Camera online" : "Camera offline");
  setStatusPill(elements.controllerStatus, status.controller_connected, status.dry_run ? "Simulation: motors disabled" : status.controller_connected ? "Drive link online" : "Drive link idle");
  elements.plannerReadout.textContent = status.planner_status || "WAITING";
  elements.commandReadout.textContent = (status.command || "stop").toUpperCase();
  elements.pathReadout.textContent = status.path_points ?? 0;
  elements.fpsReadout.textContent = Number(status.fps || 0).toFixed(1);
  elements.featuresReadout.textContent = status.tracked_features ?? 0;
  elements.inliersReadout.textContent = `${status.tracked_inliers ?? 0} inliers`;
  elements.voStatus.textContent = `${status.vo_status || "INITIALIZING"} · ${status.calibrated ? "Camera calibrated" : "Approximate camera intrinsics"}`;
  elements.perceptionAge.textContent = status.perception_age_ms == null ? "Waiting for perception" : `Perception age: ${Math.round(status.perception_age_ms)} ms`;
  elements.floorOverride.textContent = `Sky override: bottom ${Math.round((status.floor_fraction ?? 0.45) * 100)}%`;
  document.querySelectorAll(".position-unit").forEach((element) => { element.textContent = status.vo_units || "relative units"; });

  const position = status.position || [0, 0, 0];
  elements.positionX.textContent = Number(position[0] || 0).toFixed(2);
  elements.positionY.textContent = Number(position[1] || 0).toFixed(2);
  elements.positionZ.textContent = Number(position[2] || 0).toFixed(2);
  elements.emergencyButton.classList.toggle("active", state.emergency);
  elements.emergencyButton.querySelector("strong").textContent = state.emergency ? "Release emergency lock" : "Emergency stop";
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("Dashboard unavailable");
    renderStatus(await response.json());
  } catch (_error) {
    setStatusPill(elements.cameraStatus, false, "Dashboard offline");
    setStatusPill(elements.controllerStatus, false, "Drive link offline");
  } finally {
    setTimeout(pollStatus, 500);
  }
}

function updateClock() {
  elements.missionClock.textContent = new Date().toLocaleTimeString([], { hour12: false });
}

pollStatus();

updateClock();
setInterval(updateClock, 1000);

const viewButtons = [...document.querySelectorAll("[data-view]")];
viewButtons.forEach((button) => button.addEventListener("click", () => {
  viewButtons.forEach((item) => item.classList.toggle("active", item === button));
  document.querySelector("#roverStream").src = `/stream.mjpg?view=${button.dataset.view}`;
}));
const uploadButton = document.querySelector("#uploadVideo");
const videoFile = document.querySelector("#videoFile");
const uploadStatus = document.querySelector("#videoUploadStatus");
uploadButton.addEventListener("click", () => videoFile.click());
videoFile.addEventListener("change", async () => {
  const file = videoFile.files[0];
  if (!file) return;
  if (file.size > 512 * 1024 * 1024) { showToast("Maximum video size is 512 MB"); return; }
  stopHeartbeat();
  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading video…";
  try {
    const response = await fetch("/api/upload", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) }, body: file });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Upload failed");
    uploadStatus.textContent = `${file.name} · test mode · motors disabled`;
  } catch (error) { uploadStatus.textContent = error.message; }
  finally { uploadButton.disabled = false; videoFile.value = ""; }
});
document.querySelector("#liveCamera").addEventListener("click", async () => {
  stopHeartbeat();
  try { await api("/api/source/live", {}); uploadStatus.textContent = "Camera source · motors disabled after test mode"; }
  catch (error) { showToast(error.message); }
});
document.querySelector("#resetVO").addEventListener("click", () => api("/api/vo/reset", {}).catch((error) => showToast(error.message)));
async function openTrackingSession() {
  try { await api("/api/session/start", {}); }
  catch (error) { showToast(error.message); setTimeout(openTrackingSession, 2000); }
}
openTrackingSession();
setInterval(() => api("/api/session/heartbeat", {}).catch(() => {}), 2000);
