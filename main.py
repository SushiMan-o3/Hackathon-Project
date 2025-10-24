import cv2 as cv
import numpy as np
import math
import mediapipe as mp

def getAngle():
    pass

import cv2 as cv
import numpy as np
from typing import Optional, Tuple, Sequence

def landmark_to_pixel_xy(landmarks: Sequence, landmark_index: int, 
                         image_width: int, image_height: int, 
                         min_visibility: float = 0.4) -> Optional[Tuple[int, int]]:
    """
    Convert a MediaPipe NORMALIZED landmark (x,y in [0,1]) into integer pixel coords (x_px, y_px).
    Returns None if the landmark's visibility is below min_visibility.
    """
    p = landmarks[landmark_index]
    vis = getattr(p, "visibility", 0.0) or 0.0
    if vis < min_visibility:
        return None
    else:
        # map normalized → pixel, clamped to image bounds
        x_px = int(np.clip(p.x * image_width,  0, image_width  - 1))
        y_px = int(np.clip(p.y * image_height, 0, image_height - 1))
        return (x_px, y_px)


def draw_joint_point(image_bgr: np.ndarray, point_xy: Optional[Tuple[int, int]],
                     bgr_color: Tuple[int, int, int], radius: int = 8, 
                     label_text: Optional[str] = None) -> None:
    """
    Draw a filled circle at a joint location, and an optional text label next to it.
    Does nothing if point_xy is None.
    """
    if point_xy is None:
        return
    else:
        cv.circle(image_bgr, point_xy, radius, bgr_color, thickness=-1)

        if label_text:
            cv.putText(image_bgr, label_text,(point_xy[0] + 8, point_xy[1] - 8),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv.LINE_AA )


def draw_line_segment(image_bgr: np.ndarray, point_a_xy: Optional[Tuple[int, int]],
                      point_b_xy: Optional[Tuple[int, int]], bgr_color: Tuple[int, int, int],
                      thickness: int = 6) -> None:
    """
    Draw a straight line between two points. Skips if either point is None.
    """
    if point_a_xy is None or point_b_xy is None:
        return
    else:
        cv.line(image_bgr, point_a_xy, point_b_xy, bgr_color, thickness, cv.LINE_AA)



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