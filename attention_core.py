
import cv2
import numpy as np
import mediapipe as mp
import time

# --- MediaPipe Setup ---
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Global Attention State ---
total_focused_time = 0
total_distracted_time = 0
last_check_time = time.time()
eye_closure_start_time = None
EYE_CLOSED_TIMER_THRESHOLD = 3  # seconds

def analyze_attention(frame):
    global total_focused_time, total_distracted_time, last_check_time, eye_closure_start_time

    current_time = time.time()
    time_elapsed = current_time - last_check_time
    last_check_time = current_time

    status = "Unknown"
    eye_closed_duration = 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_detection = face_detector.process(rgb_frame)
    results_mesh = face_mesh_detector.process(rgb_frame)

    if results_detection.detections and results_mesh.multi_face_landmarks:
        face_landmarks = results_mesh.multi_face_landmarks[0].landmark

        # ----- EYE CLOSURE DETECTION -----
        left_eye_top = face_landmarks[159]
        left_eye_bottom = face_landmarks[145]
        right_eye_top = face_landmarks[386]
        right_eye_bottom = face_landmarks[374]

        def get_eye_ratio(top, bottom):
            return abs(top.y - bottom.y)

        left_eye_ratio = get_eye_ratio(left_eye_top, left_eye_bottom)
        right_eye_ratio = get_eye_ratio(right_eye_top, right_eye_bottom)
        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if avg_eye_ratio < 0.01:
            if eye_closure_start_time is None:
                eye_closure_start_time = current_time
            eye_closed_duration = current_time - eye_closure_start_time
            if eye_closed_duration >= EYE_CLOSED_TIMER_THRESHOLD:
                total_distracted_time += time_elapsed
                status = "Eyes closed too long ❌"
            else:
                total_focused_time += time_elapsed
                status = "Blinking / Partial Attention"
        else:
            eye_closure_start_time = None

            # ----- GAZE/HEAD TURN DETECTION -----
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            nose_tip = face_landmarks[1]

            eye_center_x = (left_eye.x + right_eye.x) / 2
            face_direction = nose_tip.x - eye_center_x

            if abs(face_direction) > 0.04:
                total_distracted_time += time_elapsed
                status = "Looking Away ❌"
            else:
                total_focused_time += time_elapsed
                status = "Attentive ✅"
    else:
        total_distracted_time += time_elapsed
        status = "No face detected ❌"

    return {
        "status": status,
        "eye_closed_seconds": round(eye_closed_duration, 2),
        "total_focused_seconds": int(total_focused_time),
        "total_distracted_seconds": int(total_distracted_time)
    }
