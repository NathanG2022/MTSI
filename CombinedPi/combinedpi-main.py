import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Counters for eye closure duration
left_eye_closed_counter = 0
right_eye_closed_counter = 0
closed_eye_frames_threshold = 15  # Adjust this threshold based on your needs

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    # Compute the Euclidean distances between the vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(np.array([eye_top.x, eye_top.y]) - np.array([eye_bottom.x, eye_bottom.y]))
    B = np.linalg.norm(np.array([eye_left.x, eye_left.y]) - np.array([eye_right.x, eye_right.y]))

    # Compute the eye aspect ratio
    ear = A / B
    return ear

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1]])
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
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

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            # Measure eyelid distance using EAR
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

            # Check if eyes are closed based on EAR threshold
            ear_threshold = 0.3  # This value may need tuning

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

            cv2.putText(image, eye_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start
        if(totalTime == 0):
            totalTime = 1
        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose Estimation', image)

    key = cv2.waitKey(9)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
