import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 3D model points for head pose estimation (nose, chin, left eye left corner, right eye right corner, left mouth corner, right mouth corner)
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets of vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    # Compute the euclidean distance between the horizontal mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[12], mouth[16])
    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2.0 * D)
    return mar

def get_head_pose(shape, image):
    # 2D image points corresponding to the 3D model points
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        # Project a 3D point (0, 0, 1000.0) onto the image plane to get the head pose direction
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # Calculate yaw, pitch, roll from rotation vector
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        yaw = angles[1] * 180 / math.pi
        pitch = angles[0] * 180 / math.pi
        roll = angles[2] * 180 / math.pi

        return yaw, pitch, roll, p1, p2
    else:
        return 0, 0, 0, (0,0), (0,0)

def detect_blinks(ear, ear_threshold=0.25, consec_frames=3):
    # If EAR is below threshold, increment blink counter
    if ear < ear_threshold:
        return True
    return False

def calculate_fatigue_score(ear_avg, mar_avg, blink_count, head_pose_stability, time_window=60):
    # Normalize metrics to 0-1 scale
    ear_score = max(0, min(1, (0.3 - ear_avg) / 0.3))  # Lower EAR indicates more fatigue
    mar_score = max(0, min(1, mar_avg / 0.7))  # Higher MAR indicates yawning
    blink_score = max(0, min(1, blink_count / (time_window / 10)))  # More blinks per minute
    pose_score = max(0, min(1, (180 - head_pose_stability) / 180))  # Deviation from neutral pose

    # Weighted average (adjust weights as needed)
    fatigue_score = (ear_score * 0.4 + mar_score * 0.3 + blink_score * 0.2 + pose_score * 0.1) * 100
    return fatigue_score
