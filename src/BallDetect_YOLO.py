import cv2
import numpy as np
from ultralytics import YOLO

# MODEL_PATH = "yolo11n.pt" # (fastest, lowest accuracy)
# MODEL_PATH = "yolo11s.pt"
# MODEL_PATH = "yolo11m.pt"
# MODEL_PATH = "yolo11l.pt"
MODEL_PATH = "yolo11x.pt" # (slowest, highest accuracy)
# MODEL_PATH = "yolov8n.pt"
# MODEL_PATH = "yolov8s.pt"
# MODEL_PATH = "yolov8m.pt"
# MODEL_PATH = "yolov8l.pt"
# MODEL_PATH = "yolov8x.pt"


CONF_THRES = 0.30
IMG_SIZE = 416

SPORTS_BALL_CLASS_ID = 32

# Load grayscale image
im = cv2.imread("pictures/ball_2.jpg", cv2.IMREAD_GRAYSCALE)
if im is None:
    raise FileNotFoundError("Could not read image at pictures/ball_3.jpg")

# YOLO models expect 3-channel images; convert gray -> BGR
im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

model = YOLO(MODEL_PATH)

# Run inference
result = model.predict(im_bgr, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=True)[0]

# Pick best detection (highest confidence)
if result.boxes is None or len(result.boxes) == 0:
    print("No ball detected")
    ball_center = None
else:
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()   # [N,4] in pixels: x1,y1,x2,y2
    conf = boxes.conf.cpu().numpy()   # [N]
    cls  = boxes.cls.cpu().numpy().astype(int)  # [N]
    
    box_xyxy = []
    box_cxcy = []

    for i in range(len(xyxy)):
        if cls[i] != SPORTS_BALL_CLASS_ID:
            print(f"Ignoring non-ball class {cls[i]}")
            continue

        x1, y1, x2, y2 = xyxy[i]
        # box_xyxy.append((int(x1), int(y1), int(x2), int(y2)))

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # box_cxcy.append((cx, cy))

        print(f"Ball detected at ({cx:.1f}, {cy:.1f}), conf={conf[i]:.2f}")

        cv2.rectangle(im_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(im_bgr, (int(cx), int(cy)), 4, (0, 255, 0), -1)

    cv2.imshow("Detection", im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # If your model only has 1 class (ball), you can skip class filtering.
    # best_i = int(np.argmax(conf))

    # x1, y1, x2, y2 = xyxy[best_i]
    # cx = 0.5 * (x1 + x2)
    # cy = 0.5 * (y1 + y2)
    # ball_center = (float(cx), float(cy))

    # print(f"Ball center: (cx, cy) = ({cx:.1f}, {cy:.1f}), conf={conf[best_i]:.2f}")

    # # Optional: visualize
    # cv2.rectangle(im_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # cv2.circle(im_bgr, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    # cv2.imshow("Detection", im_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
