from flask import Flask, Response, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

app = Flask(__name__)
CORS(app)

# Initialize color points for drawing
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes for different color points
blue_index = green_index = red_index = yellow_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Color settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create a canvas for drawing
paintWindow = np.zeros((600, 800, 3), dtype=np.uint8) + 255

# Initialize MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def generate_frames():
    global bpoints, gpoints, rpoints, ypoints
    global blue_index, green_index, red_index, yellow_index
    global paintWindow, colorIndex

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw UI buttons
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (50, 50, 50), -1)
        for i, color in enumerate(colors):
            x1 = 160 + i * 155
            x2 = x1 + 140
            frame = cv2.rectangle(frame, (x1, 1), (x2, 65), color, -1)
            cv2.putText(frame, ["Blue", "Green", "Red", "Yellow"][i], (x1 + 35, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "CLEAR", (49, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Hand landmark detection
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            fore_finger = (landmarks[8][0], landmarks[8][1])

            if fore_finger[1] <= 65:  # Check for button press
                if 40 <= fore_finger[0] <= 140:  # Clear button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    paintWindow[67:, :, :] = 255
                    blue_index = green_index = red_index = yellow_index = 0
                elif 160 <= fore_finger[0] <= 300:
                    colorIndex = 0  # Blue
                elif 315 <= fore_finger[0] <= 455:
                    colorIndex = 1  # Green
                elif 470 <= fore_finger[0] <= 610:
                    colorIndex = 2  # Red
                elif 625 <= fore_finger[0] <= 765:
                    colorIndex = 3  # Yellow

            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(fore_finger)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(fore_finger)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(fore_finger)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(fore_finger)

        # Draw the points
        points = [bpoints, gpoints, rpoints, ypoints]
        for i, color_point in enumerate(points):
            for j in range(len(color_point)):
                for k in range(1, len(color_point[j])):
                    if color_point[j][k - 1] is None or color_point[j][k] is None:
                        continue
                    cv2.line(frame, color_point[j][k - 1], color_point[j][k], colors[i], 2)
                    cv2.line(paintWindow, color_point[j][k - 1], color_point[j][k], colors[i], 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
