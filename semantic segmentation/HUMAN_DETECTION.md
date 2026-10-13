# Human detection

ANVESHAK now runs YOLOv8n COCO `person` detection in the main perception loop.

- Model: `yolov8n.pt`
- Class: `person` (COCO class 0)
- Default confidence: 0.50
- Default input size: 640
- Default inference interval: every frame
- Bounding boxes are drawn on the dashboard stream.
- A red dashboard alert and toast appear when a human is detected.

Put `yolov8n.pt` beside `combined_run.py`, or let Ultralytics download it automatically on first startup.

Example:
`python combined_run.py --source 0 --human-confidence 0.50 --human-every 1`
