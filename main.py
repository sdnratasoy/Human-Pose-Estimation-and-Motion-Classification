import cv2
import mediapipe as mp
import numpy as np 

def calculate_angle(a,b,c): 
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

cap = cv2.VideoCapture("squat_test.avi")

counter = 0 
stage = None

def classify_pose(knee_angle):
    if knee_angle < 100:
        return "Squatting"
    elif 100 <= knee_angle <= 160:
        return "Lunging"
    else:
        return "Standing"

with mp_pose.Pose(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # videoyu başa al
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]
            ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            ]

            angle = calculate_angle(hip, knee, ankle)
            current_pose = classify_pose(angle)

            if angle < 90:
                stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1

            cv2.putText(image, f"Diz aci: {int(angle)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (255,255,255), 2)
            cv2.putText(image, f"Squat sayisi: {counter}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.putText(image, f"Pose: {current_pose}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (0,255,255), 2)

        except:
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )

        cv2.imshow("Pose classifier ile hareket sayaci", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
