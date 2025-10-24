import cv2 as cv
import numpy as np
import math
import mediapipe as mp


def main():
    cap = cv.VideoCapture(0)

    # used for mapping out the different joints based on the library (confidence basically how 
    # confident should the library be before detecting)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) 
    mp_drawing = mp.solutions.drawing_utils # used for drawing the figures

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = mp_pose.process(frame) # runs a pose model for the frame

        mp_drawing.draw_landmarks(
            frame, # draws image on the frame itself
            results.pose_landmarks, # for each of the frame work
            mp.solutions.pose.POSE_CONNECTIONS, # for the lines connecting them all 
            mp_drawing.DrawingSpec(color=(180,180,180), thickness=4, circle_radius=2), # for the dots
            mp_drawing.DrawingSpec(color=(180,180,180), thickness=6, circle_radius=2) # for the lines
        )
        
        # Display the resulting frame
        cv.imshow('frame', frame)
        
        # quits the loop if q is pressed
        if cv.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()