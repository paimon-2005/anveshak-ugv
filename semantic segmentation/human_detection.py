import cv2

try:
    from ultralytics import YOLO
except ImportError:  # Optional until human detection is actually started.
    YOLO = None

class HumanDetector:
    PERSON_CLASS = 0

    def __init__(self, model_path="yolov8n.pt", confidence=0.50, image_size=640, device=None):
        if YOLO is None:
            raise RuntimeError("Human detection requires the ultralytics package. Install requirements.txt first.")
        self.model = YOLO(str(model_path))
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device

    def detect(self, frame):
        kwargs = {
            "source": frame, "classes": [self.PERSON_CLASS],
            "conf": self.confidence, "imgsz": self.image_size, "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        return [{"box": tuple(map(int, box)), "confidence": float(conf)} for box, conf in zip(boxes, confidences)]

    @staticmethod
    def annotate(frame, detections):
        output = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"HUMAN {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, th + 8)
            cv2.rectangle(output, (x1, label_y - th - 8), (x1 + tw + 8, label_y), (0, 0, 255), -1)
            cv2.putText(output, label, (x1 + 4, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if detections:
            message = f"HUMAN ALERT | {len(detections)} DETECTED"
            cv2.rectangle(output, (0, 44), (min(output.shape[1], 300), 78), (0, 0, 180), -1)
            cv2.putText(output, message, (12, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return output
