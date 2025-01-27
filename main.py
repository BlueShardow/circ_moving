import cv2
import os
import numpy as np

def resize_to_480p(vid):
    height, width = vid.shape[:2]
    aspect_ratio = width / height
    new_height = 480
    new_width = int(new_height * aspect_ratio)

    return cv2.resize(vid, (new_width, new_height))

def enhance_contrast(vid):
    return cv2.normalize(vid, None, 0, 255, cv2.NORM_MINMAX)

vid_path = "/Users/pl1001515/Downloads/can.mp4"

if not os.path.exists(vid_path): 
    print(f"Error: File '{vid_path}' not found.")

else:
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print("Error: Could not open video.")

    else:
        print("Video Found")

        vid_480p = resize_to_480p(vid_path)

        x, y, w, h = 2, 0, 999, 999

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame or end of video.")
                break

            cropped_frame = frame[y:y+h, x:x+w]

            vid_blur = cv2.medianBlur(vid_480p, 5)
            vid_contrast = enhance_contrast(vid_blur)

            circles = cv2.HoughCircles(vid_contrast, cv2.HOUGH_GRADIENT, 1.1, 50, param1=65, param2=90, minRadius=10, maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(vid_contrast, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(vid_contrast, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('detected circles', vid_contrast)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
