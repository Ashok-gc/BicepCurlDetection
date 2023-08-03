# importing libraries
import cv2
import numpy as np
import mediapipe as mp

# importing modules
def calculate_angle(a,b,c):
    a = np.array(a.x, a.y) # First
    b = np.array(b.x, b.y) # Mid
    c = np.array(c.x, c.y) # End

    ab = np.subtract(a,b)
    bc = np.subtract(b,c)

    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14
    return np.round(theta, 2)

# 
def infer():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    left_flag = None
    left_count = 0
    right_flag = None
    right_count = 0

    # For webcam input:
    cap = cv2.VideoCapture(0)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        _, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image, str(left_angle),
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(right_angle),
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # Counter
            if left_angle > 160:
                left_flag = True
            if right_angle > 160:

                right_flag = True

            if left_flag:
                if left_angle < 30:
                    left_count += 1
                    left_flag = False
            if right_flag:
                if right_angle < 30:
                    right_count += 1
                    right_flag = False
                    
            cv2.putText(image, str(left_count),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(right_count),
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                        )
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(10) & 0xFF == ord('r'):
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer()