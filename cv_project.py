import cv2
import numpy as np
import time 

# Path to video  
video_path = "bicycle1.mp4" 
video = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video file")
    exit()

# Global variables
x_min, y_min, x_max, y_max = 0, 0, 0, 0
drawing = False
paused = False
current_frame_copy = None # To store the clean frame for drawing resets

def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max, drawing, frame, current_frame_copy

    # Only allow drawing if the video is PAUSED
    if not paused:
        return

    # Left button down - start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_min, y_min = x, y
        x_max, y_max = x, y

    # Mouse move - update rectangle while drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_max, y_max = x, y
            # Refresh frame from the clean copy so we don't draw multiple rectangles
            frame = current_frame_copy.copy()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("coordinate_screen", frame)

    # Left button up - finish drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_max, y_max = x, y
        # Ensure min/max are correct
        if x_min > x_max: x_min, x_max = x_max, x_min
        if y_min > y_max: y_min, y_max = y_max, y_min
        
        # Draw final rectangle
        frame = current_frame_copy.copy()
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("coordinate_screen", frame)
        print(f"Rectangle selected: ({x_min}, {y_min}) to ({x_max}, {y_max})")

cv2.namedWindow('coordinate_screen')
cv2.setMouseCallback('coordinate_screen', coordinat_chooser)

print("="*60)
print("OBJECT SELECTION PHASE")
print("="*60)
print("1. Video will play automatically.")
print("2. Press 'SPACE' to PAUSE when the object appears.")
print("3. Draw the rectangle while paused.")
print("4. Press 'ENTER' to confirm selection and start tracking.")
print("="*60)

# --- NEW: Video Playback & Selection Loop ---
while True:
    if not paused:
        ret, frame = video.read()
        if not ret:
            print("End of video reached before selection. Restarting...")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Save a copy for the mouse callback to use as a "clean slate"
        current_frame_copy = frame.copy()
        
        # Display instructions on the live video
        cv2.putText(frame, "Press SPACE to Pause & Select", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("coordinate_screen", frame)
        
    else:
        # If paused, we just wait for mouse events (handled by coordinat_chooser)
        # We don't refresh the frame here to prevent flickering
        pass

    k = cv2.waitKey(30) & 0xFF
    
    if k == 27:  # ESC to Quit
        print("Exiting...")
        cv2.destroyAllWindows()
        video.release()
        exit(0)
    elif k == 32: # SPACE key to Toggle Pause
        paused = not paused
        if paused:
            print("Paused. Draw your rectangle now.")
        else:
            print("Resumed.")
    elif k == 13: # ENTER key to Confirm Selection
        if x_max - x_min > 10 and y_max - y_min > 10:
            print("Selection confirmed. Starting tracking...")
            break
        else:
            print("Cannot start: No valid rectangle drawn yet.")

# --- SELECTION DONE ---
# Important: 'frame' is now the frame where we drew the box.
# We must use THIS frame to initialize Harris Corners.

# Validate rectangle selection
if x_max - x_min < 10 or y_max - y_min < 10:
    print("Error: Invalid selection.")
    video.release()
    exit(1)

print(f"\nROI Selected: ({x_min}, {y_min}) to ({x_max}, {y_max})")

############################ STEP 1: HARRIS CORNER DETECTION ####################################

# Use the frame where selection happened (current 'frame')
first_gray = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2GRAY)

# Create a mask for the ROI (only detect corners inside the rectangle)
roi_mask = np.zeros_like(first_gray)
roi_mask[y_min:y_max, x_min:x_max] = 255

# Parameters for Shi-Tomasi corner detection
feature_params = dict(
    maxCorners=100,           
    qualityLevel=0.3,         
    minDistance=7,            
    blockSize=7,              
    useHarrisDetector=True,   
    k=0.04                    
)

# Detect corners ONLY inside the ROI using the mask
p0 = cv2.goodFeaturesToTrack(first_gray, mask=roi_mask, **feature_params)

if p0 is None or len(p0) == 0:
    print("No features found in the selected region!")
    video.release()
    exit()

print(f"Harris Corner Detection: Found {len(p0)} keypoints")

# Show keypoints briefly
keypoint_frame = current_frame_copy.copy()
for point in p0:
    x, y = point.ravel()
    cv2.circle(keypoint_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
cv2.rectangle(keypoint_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('coordinate_screen', keypoint_frame)
cv2.waitKey(1000) # Show for 1 second then continue

############################ STEP 2: TRACKING SETUP ####################################

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Initialize variables for the tracking loop
trajectory_mask = np.zeros_like(frame)
colors = np.random.randint(0, 255, (len(p0), 3))
frame_count = 0
start_time = time.time()

# IMPORTANT: old_gray must match the frame where p0 was detected
old_gray = first_gray.copy() 

print("\n" + "="*60)
print("TRACKING STARTED")
print("="*60)

############################ STEP 3: TRACKING LOOP ####################################

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video reached.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            good_new = p1[status.flatten() == 1]
            good_old = p0[status.flatten() == 1]
            
            if len(good_new) > 0:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    color = colors[i % len(colors)].tolist()
                    trajectory_mask = cv2.line(trajectory_mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)

                img = cv2.add(frame, trajectory_mask)

                # Info display
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Lucas-Kanade Optical Flow Tracking', img)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            p0 = None

    # Re-detect if lost
    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if p0 is not None:
            colors = np.random.randint(0, 255, (len(p0), 3))

    k = cv2.waitKey(30) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
video.release()