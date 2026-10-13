# Validation

`model_validation.json` records the existing segmentation model comparison. The integrated pipeline tests cover terrain-mask behavior, A* route generation, motor safety, video upload and source switching, dashboard views, and launching the isolated OpenCV visual odometry process.

Run:

```bash
python -m pytest -q
```

Native OpenCV window creation requires a graphical desktop. Automated tests replace the launched process with a test double and verify its command, selected input source, duplicate-launch guard, and shutdown handling without opening a GUI.
