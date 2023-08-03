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