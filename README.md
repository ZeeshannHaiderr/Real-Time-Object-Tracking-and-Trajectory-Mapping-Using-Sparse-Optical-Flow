# Real-Time Object Tracking and Trajectory Mapping Using Sparse Optical Flow

This project implements a robust classical computer vision framework for real-time object tracking and motion path estimation. By leveraging feature-based motion estimation and probabilistic filtering, the system achieves stable performance without the need for computationally expensive deep learning models.

---

## Core Concepts
The system is built on a foundational approach to computer vision:
1.  **Motion Estimation:** Utilizing the brightness constancy assumption to track pixel displacement between frames.
2.  **Geometric Consistency:** Applying iterative estimation to filter out erroneous motion vectors (outliers).
3.  **Probabilistic Filtering:** Using state-space modeling to predict object position and smooth noisy sensor data.

---

## System Pipeline
The project follows a multi-stage processing pipeline to ensure real-time accuracy:

* **Target Initialization:** A manual Region of Interest (ROI) selection mechanism allows for precise target definition and minimizes background interference.
* **Feature Detection:** The **Shi-Tomasi Corner Detector** identifies salient points within the ROI that are suitable for tracking across consecutive frames.
* **Sparse Optical Flow:** The **Pyramidal Lucas-Kanade** method estimates the displacement of these feature points, handling moderate object motion efficiently.
* **Outlier Rejection:** A **RANSAC-based** mechanism eliminates inconsistent motion vectors caused by noise or dynamic backgrounds.
* **Motion Modeling:** A **Kalman Filter** (Constant Velocity Model) is integrated to suppress trajectory jitter and provide continuous state estimation even under moderate noise.
* **Trajectory Visualization:** The system accumulates filtered positions to map and display the complete motion path of the target object.

---

## Technical Specifications
* **Framework:** Classical Computer Vision (OpenCV).
* **Algorithm:** Lucas-Kanade Sparse Optical Flow.
* **Filtering:** Linear Kalman Filter.
* **Validation:** RANSAC (Random Sample Consensus).
* **Performance:** Real-time execution on standard computing hardware without training overhead.

---

## Key Features
* **Robustness:** Maintains object lock under partial occlusions and illumination variations.
* **Smoothness:** Kalman filtering significantly reduces trajectory jitter compared to raw flow estimates.
* **Efficiency:** Designed for resource-constrained environments where deep learning-based trackers are impractical.
* **Modular Design:** The architecture allows individual components (like the filter or detector) to be easily upgraded or replaced.

---

## Mathematical Foundation
The tracking logic is grounded in the optical flow constraint equation:

I_{x}u + I_{y}v + I_{t} = 0

Where:
* (u, v) represents pixel displacement.
* I_{x}, I_{y} are spatial image gradients.
* I_{t} is the temporal image gradient.

The Kalman Filter further refines this by modeling the system state x_k as:

x_{k} = [x, y, v_{x}, v_{y}]^{T}

---

## Conclusion
This project demonstrates that classical computer vision techniques remain highly effective for real-time tracking applications. By combining spatial feature tracking with temporal filtering, the framework provides a reliable and interpretable solution for surveillance, robotics, and navigation tasks.
