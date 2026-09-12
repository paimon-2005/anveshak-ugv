# ANVESHAK web dashboard

Run the existing pipeline normally:

```bash
python3 combined_run.py
```

For full setup, calibration, floor-override, and validation instructions, see `../README.md`.

The dashboard is available at `http://localhost:8080`. From another device on
the rover's network, open port `8080` at the computer's local IP address.

The pipeline starts in **manual mode** and sends a stop command whenever the
mode changes. Manual directional input always takes drive authority away from
autonomous mode. A dead-man timer also stops the rover if a held control stops
sending heartbeats for 650 ms.

## ESP32 motor command endpoint

The default request format is:

```text
http://192.168.4.1/action?go={command}&speed={speed}
```

If the rover firmware uses a different endpoint, set
`ROVER_COMMAND_URL_TEMPLATE`. The template must contain `{command}` and may
contain `{speed}`.

Example:

```bash
export ROVER_COMMAND_URL_TEMPLATE='http://192.168.4.1/control?state={command}&speed={speed}'
python3 combined_run.py
```

The default command values are `forward`, `backward`, `left`, `right`, and
`stop`. Firmware that expects single-letter commands can be configured without
changing the code:

```bash
export ROVER_FORWARD_VALUE=F
export ROVER_BACKWARD_VALUE=B
export ROVER_LEFT_VALUE=L
export ROVER_RIGHT_VALUE=R
export ROVER_STOP_VALUE=S
python3 combined_run.py
```

Optional settings:

- `DASHBOARD_PORT=8080` changes the dashboard port.
- `DASHBOARD_HOST=0.0.0.0` changes its listening address.
- `SHOW_LOCAL_WINDOW=1` restores the original OpenCV desktop window alongside
  the web dashboard.

Keep the dashboard on the rover's trusted local network. It intentionally has
no public deployment or remote-access configuration.

Autonomous commands require fresh perception and also have a dead-man timeout. Use `--dry-run` to disable motor requests. Video files always run with motors disabled. The ESP32 firmware must have its own heartbeat timeout.

The website includes All views, Only VO, Only Segmentation, Only A*, Upload test video, Return to camera, and Reset VO controls. VO starts on page open. See the Dashboard update section in the README for shared-session and test-mode behavior.
