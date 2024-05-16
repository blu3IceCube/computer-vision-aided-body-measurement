import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

def speak(audio):

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate',150)

    engine.setProperty('voice', voices[0].id)
    engine.say(audio)

    # Blocks while processing all the currently
    # queued commands
    engine.runAndWait()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    enable_segmentation=True,
                    smooth_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Known height of the person in centimeters
person_height_cm = 162

# Minimum and maximum ratio for person's height in frame
min_body_ratio = 0.85
max_body_ratio = 1.15

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to get pose landmarks
    results = pose.process(image)

    # Convert the RGB image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if all necessary landmarks are detected
        if all(landmark for landmark in [
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]]):

            # Calculate person's height and image height ratio
            person_height_pixels = np.linalg.norm(
                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x - results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y - results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            ) * image.shape[1]  # Multiply by image width to convert to absolute pixels
            image_height = image.shape[0]
            body_ratio = person_height_pixels / image_height

            # Check if person is within good frame range
            if min_body_ratio < body_ratio < max_body_ratio:
                # Calculate shoulder width
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_distance_pixels = np.linalg.norm(
                    [left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y]
                ) * image.shape[1]  # Multiply by image width to convert to absolute pixels
                shoulder_width_cm = (shoulder_distance_pixels / person_height_pixels) * person_height_cm
                real_shoulder_width = shoulder_width_cm

                # Display shoulder width
                cv2.putText(image, f'Shoulder Width: {real_shoulder_width:.2f} cm',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                speak(f' Your Shoulder Width is {real_shoulder_width:.2f} cm')
            else:
                # Display guidance message based on body ratio
                if body_ratio < min_body_ratio:
                    cv2.putText(image, 'Move closer to fill the frame', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                elif body_ratio > max_body_ratio:
                    cv2.putText(image, 'Move back slightly', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    

        # No body landmarks detected or body not in good frame range, skip to next frame
        else:
            continue

    # Display the image
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release resources
pose.close()
cap.release()
cv2.destroyAllWindows()