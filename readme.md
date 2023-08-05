# Bicep Curl Detection and Counter

## Overview

This Python application uses computer vision and the MediaPipe library to detect and count bicep curls. It analyzes the angles between key points on the user's body to determine the motion of the bicep curls and keeps track of the number of curls performed by each arm.

## Technologies Used

- **Python**: The application is written in Python, a versatile and widely-used programming language.

- **OpenCV**: OpenCV is used for video capture and image processing. It allows the application to access the user's camera feed and manipulate the frames.

- **NumPy**: NumPy is used for numerical computations and mathematical operations, essential for angle calculations.

- **MediaPipe**: MediaPipe is a library from Google that provides pre-trained models for pose estimation. It is used to detect key points on the user's body and track their positions.

## Features

- **Bicep Curl Detection**: The application detects the user's body key points, particularly the positions of the shoulders, elbows, and wrists, to determine bicep curl motions.

- **Angle Calculation**: It calculates the angles between the shoulder, elbow, and wrist joints to determine when a bicep curl is performed.

- **Curl Counting**: The application counts the number of bicep curls performed by each arm separately.

- **User Interaction**: The application provides user interaction through key presses. The 'S' key starts detection, 'Q' exits the application, and 'R' resets the curl counts.

- **Visualization**: It visualizes the angles and curl counts on the camera feed.

