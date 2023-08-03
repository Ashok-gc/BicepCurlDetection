import cv2
import numpy as np
import mediapipe as mp

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)

    theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    theta = 180 - 180 * theta / np.pi
    return np.round(theta, 2)

def start_detection():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    left_flag = None
    left_count = 0
    right_flag = None
    right_count = 0

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    detection_started = False

    while cap.isOpened():
        _, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not detection_started:
            cv2.putText(image, "Press 'S' key to start", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif detection_started:
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(image, str(left_angle),
                            tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(right_angle),
                            tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if left_angle > 160:
                    left_flag = 'down'
                if left_angle < 50 and left_flag == 'down':
                    left_count += 1
                    left_flag = 'up'

                if right_angle > 160:
                    right_flag = 'down'
                if right_angle < 50 and right_flag == 'down':
                    right_count += 1
                    right_flag = 'up'

            except:
                pass

            cv2.putText(image, "Press 'Q' to exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Left Hand Count: {left_count}',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Hand Count: {right_count}',
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(66, 245, 117), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(66, 245, 117), thickness=2, circle_radius=2)
                                      )

        cv2.imshow('MediaPipe feed', image)

        key = cv2.waitKey(30) & 0xff
        if key == 27:  # Press 'Q' key to exit
            break
        elif key == ord('s') and not detection_started:  # Press 'S' key to start detection
            detection_started = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_detection()
