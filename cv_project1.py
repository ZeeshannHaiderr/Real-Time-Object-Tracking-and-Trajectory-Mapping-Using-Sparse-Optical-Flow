import cv2
import numpy as np

# --- 1. KALMAN FILTER CLASS ---
class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 

    def set_initial_state(self, x, y):
        self.kf.statePost = np.array([[np.float32(x)], 
                                      [np.float32(y)], 
                                      [0], 
                                      [0]], dtype=np.float32)

    def update(self, coord):
        prediction = self.kf.predict()
        measurement = np.array([[np.float32(coord[0])], 
                                [np.float32(coord[1])]])
        self.kf.correct(measurement)
        return int(prediction[0]), int(prediction[1])

# --- CONFIGURATION ---
VIDEO_PATH = "bicycle1.mp4" 
OUTPUT_FILENAME = "tracking_result.mp4" # The file to save

feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- GLOBALS ---
drawing = False
roi_selected = False
paused = False
x_min, y_min, x_max, y_max = 0, 0, 0, 0
kalman = KalmanTracker()
kalman_path = []
current_frame = None 

def select_roi(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max, drawing, roi_selected, current_frame
    if not paused: return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_min, y_min = x, y
        x_max, y_max = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x_max, y_max = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True

# --- MAIN SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)

# --- VIDEO WRITER SETUP (NEW) ---
# We get the width, height, and FPS from the original video to ensure the output matches
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the VideoWriter
# 'mp4v' is a standard codec for .mp4 files
out = cv2.VideoWriter(OUTPUT_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

cv2.namedWindow('Advanced Tracking')
cv2.setMouseCallback('Advanced Tracking', select_roi)

print("Press SPACE to Pause -> Draw Box -> Press ENTER")

# --- SELECTION LOOP ---
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret: break
        current_frame = frame.copy()
    else:
        frame = current_frame.copy()
        if drawing or roi_selected:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "PAUSED. Draw & Press ENTER", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not paused:
        cv2.putText(frame, "Press SPACE to Pause", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Advanced Tracking', frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27: 
        # Clean exit without saving if user quits early
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        exit()
    elif key == 32: paused = not paused
    elif key == 13 and roi_selected: break 

# --- INITIALIZE TRACKING ---
old_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
roi_mask = np.zeros_like(old_gray)
roi_mask[y_min:y_max, x_min:x_max] = 255

p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

if p0 is None:
    print("No features found.")
    exit()

initial_x = np.mean(p0[:, 0, 0])
initial_y = np.mean(p0[:, 0, 1])
kalman.set_initial_state(initial_x, initial_y)

print(f"Tracking started! Recording to {OUTPUT_FILENAME}...")

# --- TRACKING LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        # RANSAC
        if len(good_new) > 4:
            M, mask_ransac = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            clean_new = good_new[mask_ransac.flatten() == 1]
        else:
            clean_new = good_new

        # KALMAN
        if len(clean_new) > 0:
            cx, cy = np.mean(clean_new, axis=0)
            kx, ky = kalman.update((cx, cy))
            kalman_path.append((kx, ky))

            # Draw Points
            for new in clean_new:
                a, b = new.ravel()
                cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

            # Draw Kalman Center
            cv2.circle(frame, (kx, ky), 5, (255, 0, 0), -1) 
            
            # Draw Path
            if len(kalman_path) > 1:
                for i in range(1, len(kalman_path)):
                    cv2.line(frame, kalman_path[i-1], kalman_path[i], (255, 0, 0), 2)

            p0 = clean_new.reshape(-1, 1, 2)

    # --- SAVE FRAME TO VIDEO ---
    out.write(frame) 

    cv2.imshow('Advanced Tracking', frame)
    old_gray = frame_gray.copy()
    
    if cv2.waitKey(30) == 27: break

# Clean Release
cap.release()
out.release() # Important: This finalizes the video file
cv2.destroyAllWindows()
print(f"Video saved successfully as {OUTPUT_FILENAME}")