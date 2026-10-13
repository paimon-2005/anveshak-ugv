import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "semantic segmentation"))

from bisenetv2 import BiSeNetV2
from segmentation import MEAN_RGB, STD_RGB, load_segmentation_model


def main():
    parser = argparse.ArgumentParser(description="Compare the original checkpoint with optimized inference")
    parser.add_argument("reference_checkpoint", type=Path)
    parser.add_argument("--video", type=Path, default=ROOT / "example.mp4")
    parser.add_argument("--output", type=Path, default=Path("model_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    reference = BiSeNetV2(4, output_aux=True).eval()
    reference.load_state_dict(torch.load(args.reference_checkpoint, map_location="cpu", weights_only=True))
    optimized = load_segmentation_model(ROOT / "semantic segmentation/bisenetv2.py", ROOT / "semantic segmentation/model_final.pth", "cpu")
    capture = cv2.VideoCapture(str(args.video))
    comparisons = []
    try:
        for frame_id in (0, 120, 600):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_id}")
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            rgb = (rgb - np.asarray(MEAN_RGB, np.float32)) / np.asarray(STD_RGB, np.float32)
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous()
            with torch.inference_mode():
                expected = reference(x)[0]
                actual = optimized(x.contiguous(memory_format=torch.channels_last))
            torch.testing.assert_close(expected, actual, atol=0.00025, rtol=0.00025)
            agreement = (expected.argmax(1) == actual.argmax(1)).float().mean().item()
            if agreement < 0.9999:
                raise AssertionError("Prediction agreement below 99.99 percent")
            comparisons.append({"frame": frame_id, "class_agreement": agreement,
                                "max_abs_logit_error": float((expected - actual).abs().max())})
    finally:
        capture.release()
    timings = {}
    with torch.inference_mode():
        for name, model, tensor in (("original", reference, x), ("optimized", optimized, x.contiguous(memory_format=torch.channels_last))):
            model(tensor)
            elapsed = []
            for _ in range(5):
                started = time.perf_counter()
                model(tensor)
                elapsed.append(time.perf_counter() - started)
            timings[name + "_median_ms"] = float(np.median(elapsed) * 1000)
    result = {"torch": torch.__version__, "opencv": cv2.__version__, "resolution": [640, 480], "cpu_threads": 2,
              "original_parameters": sum(p.numel() for p in reference.parameters()),
              "inference_parameters": sum(p.numel() for p in optimized.parameters()), "comparisons": comparisons,
              "timings": timings, "speedup_this_environment": timings["original_median_ms"] / timings["optimized_median_ms"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
