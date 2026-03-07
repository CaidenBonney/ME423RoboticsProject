import cv2

im_pos = cv2.imread("aruco_pose_estimation.png")
im_undist = cv2.imread("undistorted.png")

def on_click_fct(im):
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = im[y, x]  # OpenCV indexing: [row, col] = [y, x]
            print(f"Pixel ({x}, {y}) -> BGR=({b}, {g}, {r})  RGB=({r}, {g}, {b})")
    return on_click
        

cv2.namedWindow("Inspector")
cv2.setMouseCallback("Inspector", on_click_fct(im_pos))

cv2.namedWindow("Undistorted")
cv2.setMouseCallback("Undistorted", on_click_fct(im_undist))

while True:
    cv2.imshow("Inspector", im_pos)
    cv2.imshow("Undistorted", im_undist)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break


cv2.destroyAllWindows()
