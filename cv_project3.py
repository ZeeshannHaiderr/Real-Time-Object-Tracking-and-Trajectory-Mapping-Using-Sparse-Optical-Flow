import cv2
import numpy as np

# --- 1. KALMAN SMOOTHER ---
class KalmanSmoother:
    def __init__(self, init_box):
        self.kf = cv2.KalmanFilter(4, 2)
        # Standard Constant Velocity Model
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05 # Tuned for CSRT
        
        x, y, w, h = init_box
        cx, cy = x + w/2, y + h/2
        self.kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]], dtype=np.float32)

    def update(self, center):
        cx, cy = center
        prediction = self.kf.predict()
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        return int(prediction[0]), int(prediction[1])
    
    def predict(self):
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

# --- CONFIGURATION ---
VIDEO_PATH = "bowler_video1.mp4" 

# --- MAIN SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('CSRT Pro Tracker')

# Use CSRT (Most Accurate Classical Tracker)
tracker = cv2.TrackerCSRT_create()

print("------------------------------------------------")
print("STEP 1: Press SPACE to select object.")
print("STEP 2: Select precise box (Exclude background).")
print("STEP 3: Press ENTER twice.")
print("------------------------------------------------")

roi_selected = False
kalman = None
path = []
frame_template = None # Used for recovery

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- PHASE 1: SELECTION ---
    if not roi_selected:
        cv2.putText(frame, "Press SPACE to Select", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('CSRT Pro Tracker', frame)
        
        k = cv2.waitKey(30) & 0xFF
        if k == 32: # SPACE
            # Select ROI
            init_box = cv2.selectROI('CSRT Pro Tracker', frame, fromCenter=False, showCrosshair=True)
            if init_box[2] > 0: 
                tracker.init(frame, init_box)
                kalman = KalmanSmoother(init_box)
                
                # Save a "Template" of the object for recovery if lost
                x,y,w,h = [int(v) for v in init_box]
                frame_template = frame[y:y+h, x:x+w]
                
                roi_selected = True
        elif k == 27: break
        continue

    # --- PHASE 2: TRACKING ---
    success, box = tracker.update(frame)

    if success:
        # 1. Tracking Success
        x, y, w, h = [int(v) for v in box]
        cx, cy = int(x + w/2), int(y + h/2)
        
        # 2. Smooth with Kalman
        kx, ky = kalman.update((cx, cy))
        path.append((kx, ky))
        
        # Update the visual box to match Kalman Center (Smooth Look)
        # Centering the box on the smooth coordinate
        smooth_x = int(kx - w/2)
        smooth_y = int(ky - h/2)

        # Draw
        cv2.rectangle(frame, (smooth_x, smooth_y), (smooth_x + w, smooth_y + h), (0, 255, 0), 2)
        cv2.putText(frame, "CSRT: Locked", (smooth_x, smooth_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Keep template updated (optional: helps with changing appearance)
        # frame_template = frame[y:y+h, x:x+w]

    else:
        # --- PHASE 3: RECOVERY (Failsafe) ---
        cv2.putText(frame, "LOST! Recovering...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Predict where it *should* be using Kalman velocity
        pred_cx, pred_cy = kalman.predict()
        
        # Search only near the predicted location (Optimization)
        search_radius = 100
        h_frame, w_frame = frame.shape[:2]
        
        y1 = max(0, pred_cy - search_radius)
        y2 = min(h_frame, pred_cy + search_radius)
        x1 = max(0, pred_cx - search_radius)
        x2 = min(w_frame, pred_cx + search_radius)
        
        search_area = frame[y1:y2, x1:x2]

        if search_area.shape[0] > frame_template.shape[0] and search_area.shape[1] > frame_template.shape[1]:
            # Template Matching
            res = cv2.matchTemplate(search_area, frame_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # If match is strong enough (Confidence > 0.6)
            if max_val > 0.6:
                # Re-initialize Tracker at new location
                found_x = x1 + max_loc[0]
                found_y = y1 + max_loc[1]
                h_t, w_t = frame_template.shape[:2]
                
                new_box = (found_x, found_y, w_t, h_t)
                tracker = cv2.TrackerCSRT_create() # Reset tracker
                tracker.init(frame, new_box)
                cv2.rectangle(frame, (found_x, found_y), (found_x+w_t, found_y+h_t), (0, 255, 255), 2) # Yellow box = Recovered

    # Draw Trajectory
    if len(path) > 1:
        cv2.polylines(frame, [np.array(path, dtype=np.int32)], False, (0, 0, 255), 2)

    cv2.imshow('CSRT Pro Tracker', frame)
    if cv2.waitKey(30) == 27: break

cap.release()
cv2.destroyAllWindows()