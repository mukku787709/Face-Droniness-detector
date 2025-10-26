import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from utils import eye_aspect_ratio, mouth_aspect_ratio, get_head_pose, detect_blinks, calculate_fatigue_score

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Indices for left and right eyes and mouth (MediaPipe landmarks)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Thresholds
EAR_THRESH = 0.25
MAR_THRESH = 0.5
CONSEC_FRAMES = 3

# Counters
blink_counter = 0
total_blinks = 0
yawn_counter = 0
total_yawns = 0
frame_counter = 0

# Lists to store metrics for fatigue scoring
ear_list = []
mar_list = []
pose_list = []

# Create logs directory if not exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initialize CSV log
csv_file = open('logs/drowsiness_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'EAR', 'MAR', 'Blink Count', 'Yawn Count', 'Head Yaw', 'Head Pitch', 'Head Roll', 'Fatigue Score'])

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye and mouth landmarks
            left_eye_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]] for i in LEFT_EYE])
            right_eye_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]] for i in RIGHT_EYE])
            mouth_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]] for i in MOUTH])

            # Calculate EAR and MAR
            leftEAR = eye_aspect_ratio(left_eye_points)
            rightEAR = eye_aspect_ratio(right_eye_points)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth_points)

            # Detect blinks
            if detect_blinks(ear, EAR_THRESH):
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    total_blinks += 1
                blink_counter = 0

            # Detect yawns
            if mar > MAR_THRESH:
                yawn_counter += 1
            else:
                if yawn_counter >= CONSEC_FRAMES:
                    total_yawns += 1
                yawn_counter = 0

            # Get head pose (using converted landmarks)
            shape = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in face_landmarks.landmark])
            yaw, pitch, roll, p1, p2 = get_head_pose(shape, frame)

            # Store metrics
            ear_list.append(ear)
            mar_list.append(mar)
            pose_list.append(abs(yaw) + abs(pitch) + abs(roll))  # Sum of absolute angles as stability measure

            # Keep only last 30 seconds of data (assuming 30 FPS)
            if len(ear_list) > 900:
                ear_list.pop(0)
                mar_list.pop(0)
                pose_list.pop(0)

            # Calculate fatigue score every second
            if frame_counter % 30 == 0 and len(ear_list) > 0:
                ear_avg = np.mean(ear_list)
                mar_avg = np.mean(mar_list)
                pose_stability = np.mean(pose_list)
                fatigue_score = calculate_fatigue_score(ear_avg, mar_avg, total_blinks, pose_stability)

                # Log to CSV
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([timestamp, ear, mar, total_blinks, total_yawns, yaw, pitch, roll, fatigue_score])

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Draw head pose line
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Yawns: {total_yawns}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if 'fatigue_score' in locals():
                cv2.putText(frame, f"Fatigue: {fatigue_score:.1f}%", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
