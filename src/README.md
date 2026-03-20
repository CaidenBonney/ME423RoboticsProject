# Source Code Reference

This folder contains the runtime code for the ME423 Robotics Project.

## Runtime Requirements

- Python `3.12.10`
- `numpy`
- `opencv-python`
- `pyrealsense2`
- `scipy`
- Quanser Python packages, including `hal.products.qarm`
- `../QarmHardwareFiles/AdjustedQarmHardwareFiles/qarm.py`

## Current `src` Folder Contents

- `main.py`: entry point that starts the camera thread, arm thread, and live visualization loop.
- `Camera.py`: RealSense setup, camera calibration, ball detection, and camera-to-robot coordinate transforms.
- `Arm.py`: QArm interface, interception logic, inverse kinematics, and motion command validation.
- `Ballistic.py`: physics-based ballistic estimator used by the active interception path.
- `Trajectory.py`: legacy/support trajectory fitting and buffering utilities.
- `camera_calib.yml`: stored OpenCV calibration parameters loaded during camera initialization.
- `README.md`: source-level technical reference for the files in this directory.

The runtime pipeline combines:

- `Camera.py` to detect the ball and convert pixel/depth data into robot-base coordinates.
- `Arm.py` to predict an interception point and command the Quanser QArm.
- `Ballistic.py` to fit a physics-based projectile model.
- `Trajectory.py` to store and fit a simpler polynomial trajectory model used mainly as legacy/support code.
- `main.py` to connect the camera thread, arm thread, queueing, and on-screen visualization.

The active pipeline is:

1. `main.py` creates a `Camera` and starts a camera worker thread.
2. `Camera.capture_and_process()` returns ball position `[x, y, z]` in the robot base frame.
3. `main.py` pushes the latest ball position into a single-item queue.
4. The arm worker creates an `Arm` object and calls `Arm.ballXYZ_to_phi_cmd_ballistic(...)`.
5. `Arm.py` uses `BallisticInterceptor` from `Ballistic.py` to predict where the ball will cross the catch plane.
6. `Arm.py` runs inverse kinematics and sends the joint command to the Quanser arm.
7. `main.py` overlays ball state, commanded arm state, and predicted trajectory on the live camera image.

## File-by-File Description

## `main.py`

`main.py` is the top-level orchestration file. It does not implement ball detection or robot kinematics itself. Its job is to coordinate the other modules and keep the camera processing and arm control running in parallel.

### Main responsibilities

- Defines lightweight data containers used for thread-safe visualization:
  - `CameraSnapshot` stores the latest frame, detected ball position, detection metadata, and timestamp.
  - `ArmOverlayState` stores the latest arm command, predicted future ball points, past measured points, and interception point.
- Defines two thread-safe wrappers:
  - `SharedLatest` stores the latest object under a lock.
  - `LatestQueue` stores only the newest unread ball measurement so stale measurements are dropped.
- Starts two worker threads:
  - `camera_worker(...)`
  - `arm_worker(...)`
- Displays the live overlay window in the main thread using OpenCV.

### Important functions

`project_robot_point_to_camera(cam, xyz_robot)`
- Converts a point from robot-base coordinates back into image coordinates using `Camera.T_RobotBase_to_Camera(...)`.
- Used only for drawing predictions and interception markers on the video feed.

`draw_camera_overlay(...)`
- Draws the detected ball center and numeric detection/debug text.
- This helper exists, but the current `main()` loop shows only the arm overlay window, not a separate camera-only window.

`draw_arm_overlay(...)`
- Draws:
  - latest ball coordinates
  - latest joint command
  - latest commanded end-effector position
  - timestamp
  - whether a trajectory fit is considered valid
  - future predicted ball points
  - past observed points
  - the selected interception point
- This function is the main visual debugging tool for the whole project.

`camera_worker(...)`
- Calls `cam.capture_and_process()` continuously.
- Converts the result into a `CameraSnapshot`.
- Publishes the snapshot to `latest_cam_snapshot`.
- Pushes the latest `(ballXYZ, ball_found, timestamp)` tuple into `ballXYZ_queue`.
- If the queue is full, the old value is discarded so the arm always works on the newest observation.

`arm_worker(...)`
- Constructs an `Arm` object.
- Reads ball measurements from `ballXYZ_queue`.
- Calls `arm.ballXYZ_to_phi_cmd_ballistic(...)`, which is the active interception strategy.
- Extracts state from `arm.ballistic_interceptor` for visualization.
- Sends the resulting command to the hardware with `arm.move(...)`.
- Terminates the QArm cleanly when the loop ends.

`manual_control_arm_worker(...)`
- Alternative arm loop for manual testing.
- Lets the user type commands such as `home`, `test1`, `test2`, or explicit joint angles.
- Useful for validating the arm hardware and kinematics independent of camera tracking.
- Not used by `main()` right now.

`main()`
- Instantiates shared state, the camera, and both worker threads.
- Waits for both threads to become ready.
- Repeatedly renders the latest arm overlay into an OpenCV window named `arm_pov`.
- Stops when the user presses `Esc` or on interrupt.

### How `main.py` interacts with other files

- Imports `Camera` from `Camera.py`.
- Imports `Arm` from `Arm.py`.
- Relies on `Camera.py` to transform raw RealSense data into robot coordinates.
- Relies on `Arm.py` to choose a catch point, solve inverse kinematics, and issue the hardware command.

## `Camera.py`

`Camera.py` owns the perception side of the project. It configures the Intel RealSense D435, calibrates the camera relative to the robot through an ArUco marker, detects the ball in color/depth images, and converts the ball position into robot-base coordinates.

### Main responsibilities

- Start and manage the RealSense depth/color pipeline.
- Detect the ArUco marker and solve the camera-to-robot transform.
- Build a background subtraction model for motion filtering.
- Detect the ball in HSV color space.
- Estimate ball depth robustly from nearby depth pixels.
- Transform the detected ball from camera coordinates into robot-base coordinates.

### Key configuration constants

At the top of the file, the constants define the physical and vision setup:

- `MARKER_ID`, `MARKER_LENGTH_M`, `ARUCO_DICT` describe the calibration marker.
- `W`, `H`, `FPS` define RealSense stream settings.
- `NEIGHBOR_RADIUS_PX` and `BALL_DEPTH_RADIUS_PX` define depth filtering neighborhoods.
- `AREA_MIN`, `AREA_MAX`, `CIRC_MIN`, `SOLID_MIN`, `ASPECT_MAX` define contour quality thresholds for ball detection.
- `CALIBRATION_FILE` points to `src/camera_calib.yml`.

### `Camera` class

`Camera.__init__()`
- Initializes timing, intrinsics, transformation placeholders, detection state, and the current frame buffer.
- Defines a measured transform `ArUco2Base_Transformation` from the marker frame into the robot base frame.
- Calls `cam_setup()` immediately, so constructing `Camera()` performs hardware startup and calibration.

`cam_setup()`
- Starts the RealSense pipeline for aligned depth and color streams.
- Reads RealSense intrinsics directly from the SDK.
- Calls `cam_calibration()`.

`cam_calibration()`
- Loads the calibration matrix and distortion coefficients from `camera_calib.yml`.
- Calls `get_robot_transformation()` to solve the camera pose relative to the ArUco marker.
- Calls `create_background_model()` to warm up the background subtractor.

`get_robot_transformation()`
- Repeatedly captures frames until the configured ArUco marker is detected and a pose can be estimated.
- Uses a hybrid approach:
  - rotation from `solvePnP`
  - translation from depth measurements at the marker corners
- Stores:
  - `R_m_c`
  - `t_m_c`
  - `T_cam2ArUco`
  - `T_cam2ArUco_inv`
- This step is blocking, so the program waits here until calibration succeeds.

`create_background_model()`
- Records a short warmup video from the camera.
- Uses those frames to initialize `cv2.createBackgroundSubtractorMOG2(...)`.
- The goal is to reduce false positives from static scene elements.

`capture_image()`
- Pulls a synchronized depth/color frame set from the RealSense pipeline.
- Aligns depth to color.
- Stores the RGB frame in `self.current_frame`.

`image_processing(aligned_frames)`
- Runs the actual ball detection pipeline.
- Calls `detect_ball_center(...)` to find the best contour candidate.
- If a ball is found:
  - stores pixel center `u`, `v`
  - stores debug score terms
  - estimates depth with `robust_depth_at_pixel(...)`
  - deprojects the pixel into 3D camera coordinates with `deproject(...)`
  - transforms that 3D point into robot-base coordinates with `T_Camera_to_RobotBase(...)`
- Returns `(ballXYZ_in_robot_frame, found_ball)`.

`T_Camera_to_RobotBase(P_ball_cam)`
- Applies:
  - camera frame -> ArUco frame via `T_cam2ArUco`
  - ArUco frame -> robot base via `ArUco2Base_Transformation`
- This is the main output transform used by the rest of the system.

`T_RobotBase_to_Camera(XYZR)`
- Inverse of the above transform chain.
- Converts robot-frame points back to image pixels.
- Used by `main.py` to render future trajectory points and intercept markers on the camera feed.

`capture_and_process()`
- Wraps capture + processing into one call.
- Returns `(XYZ, ball_found, timestamp)`.
- This is the main API used by `main.py`.

### Important helper functions

`detect_ball_center(...)`
- Segments the frame by color in HSV.
- Optionally combines the color mask with background subtraction.
- Finds contours and scores each contour by:
  - circularity
  - solidity
  - bounding-box aspect ratio
- Picks the best-scoring contour and returns its centroid.
- The current call path uses `GREEN_BALL_COLOR`.

`robust_depth_at_pixel(...)`
- Samples depth values in a neighborhood around the selected pixel.
- Returns the median valid depth.
- This is important because single-pixel depth readings are noisy.

`deproject(...)`
- Uses the RealSense SDK to convert `(u, v, depth)` into a 3D point in the camera frame.

`get_camera_to_marker_transform(...)`
- Detects the marker.
- Builds corresponding 2D/3D or 3D/3D point sets.
- Solves the camera-to-marker pose.
- Falls back to the Kabsch rigid transform if the hybrid method fails.

`detect_marker_corners(...)`, `estimate_translation_from_depth_corners(...)`, `kabsch_rigid_transform(...)`, `build_T(...)`, `load_calibration(...)`
- These support the calibration step and are not called directly by the rest of the project.

### How `Camera.py` interacts with other files

- Provides `Camera.capture_and_process()` to `main.py`.
- Provides `T_RobotBase_to_Camera(...)` so `main.py` can visualize points predicted by `Arm.py`.
- Feeds the arm with ball coordinates already expressed in the robot base frame, so `Arm.py` does not need to know anything about camera intrinsics or image processing.

## `Arm.py`

`Arm.py` owns the robot-side logic. It wraps the Quanser QArm hardware API, tracks the arm state, predicts where the ball should be intercepted, solves inverse kinematics, and issues safe arm commands.

### Main responsibilities

- Initialize the QArm hardware.
- Home the arm and determine the startup offset between measured and commanded joint positions.
- Store the latest measured and commanded arm states.
- Predict an interception point from camera measurements.
- Solve inverse kinematics for the interception point.
- Enforce joint and workspace safety checks.
- Send the command to the physical arm.

### Initialization and hardware setup

`Arm.__init__()`
- Dynamically loads a modified `qarm.py` from `../QarmHardwareFiles/AdjustedQarmHardwareFiles/qarm.py`.
- Creates the physical arm object with `qarm.QArm(hardware=1)`.
- Creates `QArmUtilities()` for the Quanser kinematic constants and helper functions.
- Initializes command/state buffers for:
  - measured joint angles
  - measured end-effector position
  - commanded joint angles
  - commanded end-effector position
  - gripper and LED state
- Creates a `Trajectory()` instance and a `BallisticInterceptor(...)`.
- Defines:
  - `fixedX` fallback catch-plane x-coordinate
  - `_catch_z` active catch-plane height
- Sends the arm to home and waits for the joints to settle.
- Stores `_phi_offset`, which corrects the mismatch between startup measured angles and the arm's logical home configuration.

### Active interception method

The active method used by `main.py` is:

`ballXYZ_to_phi_cmd_ballistic(XYZ, ball_found, timestamp)`

This function:

- Handles lost tracking by counting missed frames.
- Resets the trajectory/interceptor if tracking is lost for too long.
- Converts the incoming ball measurement into a clean `(3,)` NumPy vector.
- Updates:
  - `self.ballistic_interceptor`
  - `self.traj`
- Asks the ballistic model for the predicted interception point on the catch plane.
- If no valid intercept exists yet, it currently falls back to:
  - `x = self.fixedX`
  - `y = current measured y`
  - `z = self._catch_z`
- Solves inverse kinematics for that intercept.
- Uses `_apply_phi_cmd(...)` to store the selected command and target position.

This method is the main connection point between the camera measurements and the robot command.

### Other interception methods

`ballXYZ_to_phi_cmd(...)`
- Older polynomial-fit method.
- Uses `Trajectory.py` to estimate future motion from fitted polynomials.
- Kept as legacy code and test support.
- Not the default method in `main.py`.

`ballXYZ_to_phi_cmd_no_traj_fixed_xz(...)`
- Test method that ignores full trajectory prediction and forces the ball target onto a fixed `x/z` plane.
- Useful for debugging inverse kinematics or controller behavior.

### Command and safety logic

`_resolve_ik(target_xyz)`
- Runs inverse kinematics.
- Chooses the valid IK solution closest to the current arm posture.
- Returns `None` if IK fails.

`_apply_phi_cmd(phi_cmd, intercept)`
- Stores `pos_cmd` and `prev_phi_cmd`.
- Intentionally does not write to `self.phi_cmd`.
- This avoids a logic bug where `move()` would see no command change and skip sending the command.

`move(phi_Cmd, gripper_Cmd=None, led_Cmd=None)`
- Validates command shapes and ranges.
- Rejects duplicate commands.
- Calls `limit_check(...)` and `workspace_check(...)`.
- Sends the command to the QArm with `read_write_std(...)`.
- This is the only function that actually transmits motion commands to hardware.

`limit_check(...)`
- Enforces joint angle limits for the base, shoulder, elbow, and wrist.

`workspace_check(...)`
- Uses forward kinematics to compute the end-effector position from the joint command.
- Rejects targets with `z < 0.1 m`.

`home()`
- Sends the arm to `[0, 0, 0, 0]` with a red LED state.

### Kinematics functions

`qarm_forward_kinematics(phi)`
- Computes the end-effector pose from the given joint angles using Quanser DH parameters.

`qarm_inverse_kinematics(p, gamma, phi_prev)`
- Computes up to four IK solutions for a Cartesian target.
- Returns both all solutions and the one closest to `phi_prev`.

### Properties

- `phi`: reads and offset-corrects the current joint angles.
- `pos`: computes the current end-effector position from the latest measured joints.
- `R`, `gripper`, `led`, `phi_dot`: expose current state.

### How `Arm.py` interacts with other files

- Imports `Trajectory` from `Trajectory.py`.
- Imports `BallisticInterceptor` from `Ballistic.py`.
- Receives ball positions from `main.py`.
- Returns joint commands and visualization state back to `main.py`.

## `Ballistic.py`

`Ballistic.py` contains the physics-first trajectory estimator used by the active arm control path.

### Purpose

Instead of fitting a free quadratic to every coordinate, this file assumes the ball follows projectile motion under constant gravity:

- `x(t)` and `y(t)` are linear.
- `z(t)` is quadratic with fixed acceleration `-g`.

This makes the model more physically meaningful and more stable than a generic polynomial fit.

### `BallisticInterceptor` class

This class maintains a rolling history of recent observations and fits the projectile model repeatedly.

Important fields:

- `catch_z`: the z-height where the arm wants to intercept the ball.
- `window_size`: maximum number of observations kept.
- `min_points`: number of observations required before predictions are trusted.
- `_t`, `_pos`: stored timestamps and positions.
- `_vx`, `_vy`, `_vz`, `_x0`, `_y0`, `_z0`: fitted model parameters.
- `_valid`: whether a usable fit currently exists.

Important methods:

`reset()`
- Clears the stored observations and invalidates the model.

`update(timestamp_ms, pos_m)`
- Appends a new observation.
- Keeps only the most recent `window_size` samples.
- Calls `_fit()` once enough points are available.

`_fit()`
- Fits:
  - `x(t)` with a line
  - `y(t)` with a line
  - `z(t)` by subtracting the known gravity term and fitting the remaining linear part
- This is the core of the ballistic model.

`predict_interception(now_ms)`
- Solves for the next future time the ball will cross `z = catch_z`.
- Uses `_solve_catch_plane(...)`.
- Returns the predicted `[x, y, z]` interception point or `None`.

`predict_pos(timestamp_ms)`
- Evaluates the fitted trajectory at an arbitrary future or past time.
- Used by `main.py` to draw the projected path.

`_solve_catch_plane(now_ms)`
- Solves the quadratic for the catch-plane crossing.
- Rejects roots that are in the past or too far in the future.

### Integration note

The bottom of the file also contains a standalone helper function named `ballXYZ_to_phi_cmd_ballistic(...)`. That function is effectively a prototype or integration shim. The current runtime path does not use it because `Arm.py` contains its own class method with the same purpose.

### How `Ballistic.py` interacts with other files

- Imported by `Arm.py`.
- Its prediction output determines the intercept target that the robot tries to reach.

## `Trajectory.py`

`Trajectory.py` is a simpler trajectory fitting utility. It stores recent measurements and fits low-order polynomials to estimate motion. In the current project state it is still updated by `Arm.py`, but it is not the primary source of interception commands anymore.

### Purpose

- Maintain a rolling buffer of timestamped positions.
- Fit:
  - linear `x(t)`
  - linear `y(t)`
  - concave-down quadratic `z(t)`
- Provide position and velocity prediction helpers.

### `Trajectory` class

`__init__()`
- Initializes polynomial coefficients, sample history, a fit-update counter, and a small buffer for smoothing `z` fits.

`update_trajectory(t, pos, window_size, update_freq=0)`
- Appends new samples.
- Trims the data window.
- Updates polynomial coefficients.
- Enforces a concave-down z fit through `_fit_concave_down_quadratic(...)`.
- Smooths z coefficients by batching them in `_update_pz_batch(...)`.

`predict_vel(tt)`
- Evaluates the derivative of the fitted polynomials.

`predict_pos(tt)`
- Intended to evaluate the fitted position.
- In the current code it returns the most recent measured position rather than the polynomial evaluation because the polynomial `x/y/z` lines are commented out.
- That means this file currently acts more as a historical buffer plus partial fitting utility than a full predictor.

`_fit_concave_down_quadratic(...)`
- Solves a least-squares quadratic fit for `z(t)`.
- Clamps the quadratic coefficient so it remains concave down.

`_update_pz_batch(...)`
- Averages a small batch of z-fits before updating the stored coefficients.

### How `Trajectory.py` interacts with other files

- Imported by `Arm.py`.
- Updated even during the ballistic path, mainly for continuity with older logic and debug support.
- Used more directly by the legacy `Arm.ballXYZ_to_phi_cmd(...)` method.

## Data Files

## `camera_calib.yml`

This file stores the OpenCV camera calibration parameters:

- `camera_matrix`
- `distortion_coefficients`

`Camera.py` loads it during startup in `cam_calibration()`. The current RealSense intrinsics come from the SDK, but the file is still read and preserved as part of the calibration workflow.

## Practical Summary

- If you want to understand the full system flow, start with `main.py`.
- If you want to understand how the ball is detected and transformed into robot coordinates, read `Camera.py`.
- If you want to understand how the arm decides where to move, read `Arm.py`.
- If you want to understand the active prediction math, read `Ballistic.py`.
- If you want to understand the older polynomial-based approach and data buffering, read `Trajectory.py`.

In the current implementation, the most important control path is:

`Camera.py` -> `main.py` queue/thread handoff -> `Arm.py` ballistic interception -> Quanser QArm command.
