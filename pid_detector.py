# P&ID Element Detection using YOLO v8

import cv2
import numpy as np
from ultralytics import YOLO

class PIDDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_elements(self, image_path):
        image = cv2.imread(image_path)
        results = self.model.predict(image)
        return results

    def draw_boxes(self, image, results):
        boxes = results.xyxy[0]  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return image

if __name__ == '__main__':
    detector = PIDDetector('yolov8_model.pt')  # Load your YOLO model
    image_results = detector.detect_elements('input_image.jpg')
    output_image = detector.draw_boxes(cv2.imread('input_image.jpg'), image_results)
    cv2.imshow('Detection', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()