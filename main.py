import cv2
import os
import numpy as np
import math

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 360
    new_width = int(new_height * aspect_ratio)

    return cv2.resize(frame, (new_width, new_height)), new_height, new_width

def enhance_contrast(frame):
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

def calculate_score(circle, last_circle, distance_weight, rad_weight):
    x, y, radius = circle
    last_x, last_y, last_radius = last_circle

    distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
    normalized_distance = distance / max(last_radius, 1)

    radius_deviation = abs(radius - last_radius)
    normalized_radius_deviation = radius_deviation / max(last_radius, 1)

    score = distance_weight * (1 - normalized_distance) + rad_weight * (1 - normalized_radius_deviation)

    return score

def find_can(frame, distance_weight = .4, rad_weight = .6, confidence_threshold = -1):
    global last_best_circle, time_decay_counter

    param1_range = (50, 150)
    param2_range = (50, 150)
    minRadius = 10
    maxRadius = 0
    best_circle = None
    best_score = -1
    height, width = frame.shape
    best_param1 = 0
    best_param2 = 0

    for param2 in range(param2_range[0], param2_range[1], 15):
        for param1 in range(param1_range[0], param1_range[1], 15):
            circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 50, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for circle in circles[0, :]:
                    x, y, radius = circle

                    if not (x - radius >= 0 and y - radius >= 0 and x + radius < width and y + radius < height):
                        continue

                    if last_best_circle is not None:
                        score = calculate_score((x, y, radius), last_best_circle, distance_weight, rad_weight)

                    else:
                        last_best_circle = (x - .01, y - .01, radius - .01)
                        score = calculate_score((x, y, radius), last_best_circle, distance_weight, rad_weight)

                    if score > best_score:
                        best_score = score
                        best_circle = (x, y, radius)
                        best_param1 = param1
                        best_param2 = param2

    if best_circle is not None:
        last_best_circle = best_circle
        time_decay_counter = 0

    else:
        circles = cv2.HoughCircles(masked_frame, cv2.HOUGH_GRADIENT, 1.1, 50, param1 = 50, param2 = 80, minRadius = 10, maxRadius = 0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
    
            for circle in circles[0, :]:
                x, y, radius = circle

                cv2.circle(contrast_frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(contrast_frame, (x, y), 2, (0, 0, 255), 3)

                print(f"Circle detected: Center=({x}, {y}), Radius={radius}")
        
        else:
            print("No circles detected.")

        time_decay_counter += 1

        if time_decay_counter > max_decay:
            last_best_circle = best_circle

    print("\nBest Circle:", best_circle)
    print("Best Score:", best_score)
    print("Best Param1:", best_param1)
    print("Best Param2:", best_param2)

    return best_circle

#Main _________________________________________________________________
vid_path = r"c:\Users\Owner\Downloads\IMG_3204.mov"

last_best_circle = None
time_decay_counter = 0
max_decay = 5

if not os.path.exists(vid_path): 
    print(f"Error: File '{vid_path}' not found.")

else:
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print("Error: Could not open video.")

    else:
        print("Video Found")

        set_fps = 7
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / set_fps)
        frame_counter = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame or end of video.")
                break

            frame_counter += 1

            if frame_counter % frame_interval != 0:
                continue

            print(f"Processing frame {frame_counter}")

            resized_frame, hieght, width = resize_frame(frame)

            x = int(width/8)
            y = int(hieght/8)
            w = int(width/1.35)
            h = int(hieght/1.35)

            cropped_frame = resized_frame[y:y+h, x:x+w]
            blurred_frame = cv2.medianBlur(cropped_frame, 5)
            contrast_frame = enhance_contrast(blurred_frame)
            gray_frame = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY)

            lower_gray = 100
            upper_gray = 255

            gray_mask = cv2.inRange(gray_frame, lower_gray, upper_gray)
            masked_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=gray_mask)

            """"
            # built in
            circles = cv2.HoughCircles(masked_frame, cv2.HOUGH_GRADIENT, 1.1, 50, param1 = 50, param2 = 80, minRadius = 10, maxRadius = 0)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    # Draw the outer circle
                    cv2.circle(contrast_frame, (i[0], i[1]), i[2], (255, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(contrast_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            # end of built in
            """


            best_circle = find_can(masked_frame)

            if best_circle is not None:
                x, y, radius = best_circle
                cv2.circle(contrast_frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(contrast_frame, (x, y), 2, (0, 0, 255), 3)


            cv2.imshow('detected circles', contrast_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

cv2.imshow("masked frame", masked_frame)
cv2.imshow("gray frame", gray_frame)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
