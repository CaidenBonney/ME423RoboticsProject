import cv2
import os
os.chdir("src/camera_matrix_discovery/sample_images")
cap = cv2.VideoCapture(3)
preview = cv2.namedWindow("preview")
snap_num = 11
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f"camera_snapshot_{snap_num}.png", frame)
        print(f"Snapshot {snap_num} saved.")
        snap_num += 1
    elif cv2.waitKey(1) == ord('q'):
        break