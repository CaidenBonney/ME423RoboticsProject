import cv2
import numpy as np

# Read image (color)
im = cv2.imread("pictures/ball_2.jpg", cv2.IMREAD_COLOR)

# --- 1) Convert to LAB and use L (brightness) channel ---
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Optional: increase local contrast on L
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
L = clahe.apply(L)

# Smooth a bit (keeps edges but reduces noise)
L_blur = cv2.GaussianBlur(L, (7, 7), 0)

# --- 2) Create contrast using a white top-hat filter ---
# IMPORTANT: kernel size should be a bit smaller than the ball diameter in pixels.
# Try (31,31), (41,41), (51,51) depending on your image scale.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
tophat = cv2.morphologyEx(L_blur, cv2.MORPH_TOPHAT, kernel)

# --- 3) Threshold the enhanced image ---
# Otsu works well after top-hat because background becomes near-zero.
_, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean up mask
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k2, iterations=1)   # remove specks
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2, iterations=2)   # fill holes

# --- 4) Blob detector tuned for the mask ---
params = cv2.SimpleBlobDetector_Params()

# In OpenCV, blob detector expects dark blobs by default in some versions.
# We give it a mask with white blobs, and explicitly filter by color.
params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 1000
params.maxArea = 200_000

params.filterByCircularity = True
params.minCircularity = 0.6

params.filterByInertia = True
params.minInertiaRatio = 0.1

params.filterByConvexity = True
params.minConvexity = 0.5

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(bw)

# Draw results
im_with_keypoints = cv2.drawKeypoints(
    im, keypoints, None, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("L (enhanced)", L)
cv2.imshow("tophat (contrast image)", tophat)
cv2.imshow("bw (mask)", bw)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()