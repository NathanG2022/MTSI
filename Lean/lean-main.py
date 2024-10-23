import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
from playsound import playsound
import mediapipe as mp
import numpy as np
import time

# Initialize variables and models
name = "User"
audio = "Wake Up " + name + "!" if len(name) > 0 else "Wake Up!"
engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

eye_closed_counter = 0
EYE_CLOSED_THRESHOLD = 5
SLEEPY_THRESHOLD = 0.2
left_eye_closed_counter = 0
right_eye_closed_counter = 0
closed_eye_frames_threshold = 15
ear_threshold = 0.3

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    A = np.linalg.norm(np.array([eye_top.x, eye_top.y]) - np.array([eye_bottom.x, eye_bottom.y]))
    B = np.linalg.norm(np.array([eye_left.x, eye_left.y]) - np.array([eye_right.x, eye_right.y]))
    ear = A / B
    return ear

def detect_eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_eye

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)
    results = face_mesh.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        left_eye = []
        right_eye = []

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        right_eye_ratio = detect_eye(right_eye)
        left_eye_ratio = detect_eye(left_eye)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        eye_ratio = round(eye_ratio, 2)

        if eye_ratio < SLEEPY_THRESHOLD:
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0

        if eye_closed_counter >= EYE_CLOSED_THRESHOLD:
            cv2.putText(frame, "Sleepy", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(frame, "Wake Up!", (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
            engine.say(audio)
            engine.runAndWait()
            playsound('siren.mp3')

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (lm.x * frame.shape[1], lm.y * frame.shape[0])
                        nose_3d = (lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * 3000)
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * frame.shape[1]
            cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                                   [0, focal_length, frame.shape[0] / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

            if x < 0:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            elif y > 10:
                text = "Looking Right"
            elif y < -10:
                text = "Looking Left"
            else:
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]

            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            right_eye_left = face_landmarks.landmark[263]
            right_eye_right = face_landmarks.landmark[362]

            left_ear = calculate_ear(left_eye_top, left_eye_bottom, left_eye_left, left_eye_right)
            right_ear = calculate_ear(right_eye_top, right_eye_bottom, right_eye_left, right_eye_right)

            if left_ear < ear_threshold:
                left_eye_closed_counter += 1
            else:
                left_eye_closed_counter = 0

            if right_ear < ear_threshold:
                right_eye_closed_counter += 1
            else:
                right_eye_closed_counter = 0

            if left_eye_closed_counter > closed_eye_frames_threshold and right_eye_closed_counter > closed_eye_frames_threshold:
                eye_status = "Sleeping"
            else:
                eye_status = "Eyes Open" if left_eye_closed_counter == 0 and right_eye_closed_counter == 0 else "Blinking"

            cv2.putText(frame, eye_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            cv2.putText(frame, f'FPS: {int(1 / (time.time() - start))}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

    faces = face_cascade.detectMultiScale(gray_scale)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)
        face_roi = gray_scale[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("Combined Drowsiness and Pose Detection", frame)

    if cv2.waitKey(9) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
