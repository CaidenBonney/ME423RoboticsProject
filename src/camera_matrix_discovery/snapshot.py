# import cv2
# import os
# os.chdir("src/camera_matrix_discovery/sample_images")
# cap = cv2.VideoCapture(3)
# preview = cv2.namedWindow("preview")
# snap_num = 12
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("preview", frame)
#     if cv2.waitKey(1) == ord('c'):
#         cv2.imwrite(f"camera_snapshot_{snap_num}.png", frame)
#         print(f"Snapshot {snap_num} saved.")
#         snap_num += 1
#         print("Ready for next snapshot.")
#     elif cv2.waitKey(1) == ord('q'):
#         break

import cv2
import os

# Save snapshots into: src/camera_matrix_discovery/sample_images (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "sample_images")
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(3)

snap_num = 12
snapshot_requested = False

def mouse_callback(event, x, y, flags, param):
    global snapshot_requested
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        snapshot_requested = True
    # If you prefer middle click (scroll wheel), use:
    # if event == cv2.EVENT_MBUTTONDOWN:

cv2.namedWindow("preview")
cv2.setMouseCallback("preview", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("preview", frame)

    # Take snapshot if requested by mouse click
    if snapshot_requested:
        out_path = os.path.join(SAVE_DIR, f"camera_snapshot_{snap_num}.png")
        cv2.imwrite(out_path, frame)
        print(f"Snapshot {snap_num} saved: {out_path}")
        snap_num += 1
        print("Ready for next snapshot.")
        snapshot_requested = False

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()