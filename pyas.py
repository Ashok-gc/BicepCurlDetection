import cv2
import numpy as np
import mediapipe as mp
from time import time

def calculate_angle(a, b, c):
    """
    Returns the angle at point b (a-b-c) in degrees (0-180).
    Inputs a, b, c are Mediapipe landmark objects with x, y.
    Uses atan2 on cross/dot for numerical stability.
    """
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    v1 = np.array([ax - bx, ay - by], dtype=np.float64)   # BA
    v2 = np.array([cx - bx, cy - by], dtype=np.float64)   # BC

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan

    v1 /= n1
    v2 /= n2

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    theta = np.degrees(np.arctan2(abs(cross), dot))  # 0..180
    return np.round(theta, 2)

def start_detection():
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    left_flag = None
    left_count = 0
    right_flag = None
    right_count = 0
    detection_started = False
    flip = True  # flip camera preview horizontally like a mirror

    # thresholds for curl counting (tune as needed)
    down_thresh = 160.0
    up_thresh = 50.0
    min_visibility = 0.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Optional: set a lower resolution to reduce CPU usage
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fps_t0 = time()
    fps_n = 0
    fps_val = 0.0

    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.5,
                      model_complexity=1,
                      smooth_landmarks=True) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not detection_started:
                cv2.putText(image, "Press 'S' to start", (int(0.3*w), 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # convenience
                    L = mp_pose.PoseLandmark
                    ls, le, lw = landmarks[L.LEFT_SHOULDER], landmarks[L.LEFT_ELBOW], landmarks[L.LEFT_WRIST]
                    rs, re, rw = landmarks[L.RIGHT_SHOULDER], landmarks[L.RIGHT_ELBOW], landmarks[L.RIGHT_WRIST]

                    # check visibility
                    def visible(*pts):
                        return all(getattr(p, "visibility", 0.0) >= min_visibility for p in pts)

                    left_angle = np.nan
                    right_angle = np.nan

                    if visible(ls, le, lw):
                        left_angle = calculate_angle(ls, le, lw)
                        cv2.putText(image, f"{left_angle if not np.isnan(left_angle) else '—'}",
                                    (int(le.x * w), int(le.y * h)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                        if not np.isnan(left_angle):
                            if left_angle > down_thresh:
                                left_flag = "down"
                            elif left_angle < up_thresh and left_flag == "down":
                                left_count += 1
                                left_flag = "up"

                    if visible(rs, re, rw):
                        right_angle = calculate_angle(rs, re, rw)
                        cv2.putText(image, f"{right_angle if not np.isnan(right_angle) else '—'}",
                                    (int(re.x * w), int(re.y * h)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                        if not np.isnan(right_angle):
                            if right_angle > down_thresh:
                                right_flag = "down"
                            elif right_angle < up_thresh and right_flag == "down":
                                right_count += 1
                                right_flag = "up"

                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

                # HUD
                cv2.putText(image, "Press 'Q' to exit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (173, 216, 230), 2, cv2.LINE_AA)
                cv2.putText(image, "Press 'R' to reset counts", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (173, 216, 230), 2, cv2.LINE_AA)
                cv2.putText(image, f"Left reps: {left_count}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (173, 216, 230), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right reps: {right_count}", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (173, 216, 230), 2, cv2.LINE_AA)

            # FPS (simple moving value)
            fps_n += 1
            t1 = time()
            if t1 - fps_t0 >= 0.5:
                fps_val = fps_n / (t1 - fps_t0)
                fps_t0 = t1
                fps_n = 0
            cv2.putText(image, f"FPS: {fps_val:.1f}", (w - 140, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe feed", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # Esc or 'q'
                break
            elif key == ord('s') and not detection_started:
                detection_started = True
            elif key == ord('r'):
                left_count = 0
                right_count = 0
                left_flag = None
                right_flag = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()
