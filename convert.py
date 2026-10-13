import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract video into numbered PNG frames")
    parser.add_argument("video", nargs="?", type=Path, default=Path(__file__).with_name("example.mp4"))
    parser.add_argument("--output", type=Path, default=Path("my_images"))
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be positive")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("*.png")):
        raise ValueError("Choose an empty output directory")
    capture = cv2.VideoCapture(str(args.video))
    index = count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {args.video}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % args.every == 0:
                target = args.output / f"{count:06d}.png"
                if not cv2.imwrite(str(target), frame):
                    raise IOError(f"Cannot save {target}")
                count += 1
            index += 1
    finally:
        capture.release()
    print(f"Saved {count} frames to {args.output}")


if __name__ == "__main__":
    main()
