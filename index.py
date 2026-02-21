import cv2
import mediapipe as mp
import os
import time

# 1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Check for the .mov file
video_path = "1.mov"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found in {os.getcwd()}")
    exit()

# Try opening with the MSMF backend for better .mov support on Windows
video_cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
cap = cv2.VideoCapture(0)


video_window_name = "GET BACK TO WORK"
window_open = False

# Track time looking away
away_start_time = None
AWAY_THRESHOLD = 60  # seconds


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    looking_down = False
    looking_right = False

    if results.multi_face_landmarks:
        # Landmark 4 is the tip of the nose
        nose = results.multi_face_landmarks[0].landmark[4]
        # Looking right if nose.x > 0.65
        if nose.x > 0.65:
            looking_right = True
        # Looking down if nose is low in the frame and not looking right
        if nose.y > 0.7 and not looking_right:
            looking_down = True

    current_time = time.time()

    if looking_down:
        if away_start_time is None:
            away_start_time = current_time
        away_duration = current_time - away_start_time
    else:
        away_start_time = None
        away_duration = 0

    # Only play video if looking down (not right) for more than threshold
    if looking_down and away_start_time is not None and away_duration >= AWAY_THRESHOLD:
        ret, v_frame = video_cap.read()
        # If we hit the end of the .mov, loop it
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()
        if ret and v_frame is not None:
            # v_frame = cv2.resize(v_frame, (640, 360))
            cv2.imshow(video_window_name, v_frame)
            window_open = True
    else:
        if window_open:
            try:
                cv2.destroyWindow(video_window_name)
                window_open = False
            except:
                pass

    # Basic preview so you can see the detection status
    status_color = (0, 0, 255) if looking_down else (0, 255, 0)
    cv2.putText(frame, f"LOOKING DOWN: {looking_down}  LOOKING RIGHT: {looking_right}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    if looking_down and away_start_time is not None:
        cv2.putText(frame, f"Down for: {int(away_duration)}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Webcam Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_cap.release()
cv2.destroyAllWindows()