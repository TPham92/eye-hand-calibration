import threading
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import pyrealsense2 as rs

CONFIG_PATH = Path(__file__).with_name("config.json")

with open(CONFIG_PATH, "r") as f:   
    config = json.load(f)

# ChArUco board
BOARD_SQUARES_X = config["charuco_marker"]["columns"]             # number of columns
BOARD_SQUARES_Y = config["charuco_marker"]["rows"]                # number of rows
SQUARE_LENGTH_MM = config["charuco_marker"]["square_size"]        # chessboard square size in mm
MARKER_LENGTH_MM = config["charuco_marker"]["marker_size"]        # inner ArUco marker size in mm

ARUCO_DICT_TYPE = getattr(cv2.aruco, config["charuco_marker"]["dictionary"])

# RealSense color stream config
COLOR_WIDTH = config["rs_camera"]["width"]
COLOR_HEIGHT = config["rs_camera"]["height"]
COLOR_FPS = config["rs_camera"]["fps"]

# ===================== MATH HELPERS =====================

def estimate_rigid_transform_3d(src_points, dst_points):
    """
    Estimate R, t such that:
        dst = R * src + t
    using Kabsch (no scale).
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    assert src.shape == dst.shape
    assert src.shape[0] >= 3

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    return R, t


def rotation_matrix_to_quaternion_xyzw(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([x, y, z, w], dtype=np.float64)


def invert_transform(R, t):
    """Invert p2 = R p1 + t to get p1 = R_inv p2 + t_inv."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def get_board_corners_3d(board):
    """
    Return Nx3 array of ChArUco board corner positions in board frame (mm)
    """
    pts = None
    pts = board.getChessboardCorners()
    pts = np.asarray(pts, dtype=np.float64)

    # shapes like (N,3) or (N,1,3) or (1,N,3)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 2 and pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)

    return pts


# ===================== GUI APP =====================

class HandEyeCharucoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Eye To Hand Calibration (ChArUco)")

        # --- ArUco / ChArUco setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Constructor-style CharucoBoard
        self.charuco_board = cv2.aruco.CharucoBoard(
            (BOARD_SQUARES_X, BOARD_SQUARES_Y),
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            self.aruco_dict
        )
        
        self.board_corners_3d = get_board_corners_3d(self.charuco_board)

        # RealSense
        self.pipeline = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # Latest frame and current corner 3D in camera
        self.latest_frame = None
        # corner_id -> 3D (mm) in camera_color_optical_frame, kept across frames
        self.current_corner_poses = {}

        # For zoom & pan
        self.zoom_scale = 1.0  # logical zoom (1.0 = full frame)
        self.center_x = None   # pan center in pixels (frame coords)
        self.center_y = None
        self.pan_step = 40     # pixels per pan click

        # For corner selection
        self.visible_corner_ids = []        # updated by capture thread
        self.selected_corner_id = None      # which corner to highlight / capture
        self.charuco_pixel_positions = {}   # cid -> (px, py) in last frame

        # Board capture state (freeze the board pose so selection doesn't jump)
        self.board_captured = False
        self.captured_corner_ids = []
        self.captured_corner_poses = {}     # cid -> 3D (mm) in camera frame
        self.captured_pixel_positions = {}  # cid -> (px, py) in captured frame

        # Status text for the UI above the dropdown
        self.board_status_var = tk.StringVar(value="✗ Board not captured")

        # Samples (camera corner vs robot base point)
        self.samples = []

        self.running = True
        self.capture_thread = None

        # Build UI
        self._build_ui()

        # Init RealSense
        self._init_realsense()

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # UI update
        self._update_image()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ==== LEFT: Video ====
        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # ==== RIGHT: Controls ====
        ctrl = ttk.Frame(main)
        ctrl.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # ---------- ChArUco board section ----------
        charuco_frame = ttk.LabelFrame(ctrl, text="ChArUco board")
        charuco_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            charuco_frame,
            text=(
                "Make sure the ChArUco board is fully visible in the color image, "
                "then click \"Capture board\" to freeze its pose for calibration."
            ),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))

        # Board status label
        self.board_status_label = tk.Label(
            charuco_frame,
            textvariable=self.board_status_var,
            fg="red",
            anchor="w",
        )
        self.board_status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 2))

        # Z-offset + Capture board row
        capture_row = ttk.Frame(charuco_frame)
        capture_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 6))

        ttk.Label(capture_row, text="Z offset (mm):").pack(side=tk.LEFT)

        self.z_offset_var = tk.StringVar(value="0.00")
        self.z_offset_entry = ttk.Entry(capture_row, textvariable=self.z_offset_var, width=8)
        self.z_offset_entry.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Button(
            capture_row,
            text="Capture board",
            command=self.on_capture_board
        ).pack(side=tk.LEFT)

        # Robot input (left) and Camera control (right)
        top_frame = ttk.Frame(ctrl)
        top_frame.pack(fill=tk.X, pady=(0, 8))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # --- Robot input frame ---
        input_frame = ttk.LabelFrame(top_frame, text="Robot input (base frame, mm)")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # Corner dropdown
        ttk.Label(input_frame, text="ChArUco Corner:").grid(row=0, column=0, sticky="e")

        self.corner_id_var = tk.StringVar()
        self.corner_id_combo = ttk.Combobox(
            input_frame,
            textvariable=self.corner_id_var,
            state="disabled",   # stays disabled until board is captured
            width=10
        )
        self.corner_id_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.corner_id_combo["values"] = []
        self.corner_id_combo.bind("<<ComboboxSelected>>", self.on_corner_selected)

        # Robot coordinates
        ttk.Label(input_frame, text="Robot X (mm):").grid(row=1, column=0, sticky="e")
        self.robot_x_entry = ttk.Entry(input_frame, width=10)
        self.robot_x_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(input_frame, text="Robot Y (mm):").grid(row=2, column=0, sticky="e")
        self.robot_y_entry = ttk.Entry(input_frame, width=10)
        self.robot_y_entry.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(input_frame, text="Robot Z (mm):").grid(row=3, column=0, sticky="e")
        self.robot_z_entry = ttk.Entry(input_frame, width=10)
        self.robot_z_entry.grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Button(input_frame, text="Capture Sample", command=self.on_capture_sample) \
            .grid(row=4, column=0, columnspan=2, pady=4)

        # --- Camera control frame ---
        camera_frame = ttk.LabelFrame(top_frame, text="Camera control")
        camera_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Zoom controls
        zoom_frame = ttk.Frame(camera_frame)
        zoom_frame.pack(pady=4)
        ttk.Button(zoom_frame, text="Zoom -", command=self.on_zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom +", command=self.on_zoom_in).pack(side=tk.LEFT, padx=2)

        # Pan controls
        pan_frame = ttk.Frame(camera_frame)
        pan_frame.pack(pady=4)

        btn_up = ttk.Button(pan_frame, text="Up", command=self.on_pan_up)
        btn_down = ttk.Button(pan_frame, text="Down", command=self.on_pan_down)
        btn_left = ttk.Button(pan_frame, text="Left", command=self.on_pan_left)
        btn_right = ttk.Button(pan_frame, text="Right", command=self.on_pan_right)

        # Arrange in a cross layout
        btn_up.grid(row=0, column=1, padx=2, pady=2)
        btn_left.grid(row=1, column=0, padx=2, pady=2)
        btn_right.grid(row=1, column=2, padx=2, pady=2)
        btn_down.grid(row=2, column=1, padx=2, pady=2)

        # ---------- Samples list ----------
        samples_frame = ttk.LabelFrame(ctrl, text="Samples")
        samples_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.samples_list = tk.Listbox(samples_frame, height=8)
        self.samples_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---------- Bottom controls (Calibrate + Save) ----------
        bottom = ttk.Frame(ctrl)
        bottom.pack(fill=tk.X)

        self.calibrate_button = ttk.Button(bottom, text="Calibrate", command=self.on_calibrate)
        self.calibrate_button.pack(side=tk.LEFT, padx=(0, 4))

        self.save_button = ttk.Button(bottom, text="Save JSON", command=self.on_save_json, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 4))

        # ---------- Result text ----------
        self.result_text = tk.Text(ctrl, height=12, wrap="word")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    # ---------- Zoom & Pan handlers ----------

    def on_zoom_in(self):
        # Zoom in (up to 3x)
        self.zoom_scale = min(self.zoom_scale * 1.25, 3.0)

    def on_zoom_out(self):
        # Zoom out, but never below 1.0 (full frame)
        self.zoom_scale = max(self.zoom_scale / 1.25, 1.0)

    def _ensure_center_initialized(self, frame_rgb):
        h, w, _ = frame_rgb.shape
        if self.center_x is None or self.center_y is None:
            self.center_x = w // 2
            self.center_y = h // 2

    def _pan(self, dx, dy):
        if self.latest_frame is None:
            return

        frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        self._ensure_center_initialized(frame_rgb)

        self.center_x += dx
        self.center_y += dy

        # We'll clamp in _update_image when cropping

    def on_pan_left(self):
        self._pan(-self.pan_step, 0)

    def on_pan_right(self):
        self._pan(self.pan_step, 0)

    def on_pan_up(self):
        self._pan(0, -self.pan_step)

    def on_pan_down(self):
        self._pan(0, self.pan_step)

    # ---------- Corner selection handler ----------

    def on_corner_selected(self, event=None):
        raw = self.corner_id_var.get().strip()
        if raw.upper().startswith("C"):
            raw = raw[1:]
        try:
            cid = int(raw)
            self.selected_corner_id = cid
        except ValueError:
            self.selected_corner_id = None

    # ---------- Board capture ----------

    def on_capture_board(self):
        """
        Capture (or recapture) the current ChArUco board pose.
        """
        if not self.current_corner_poses or not self.charuco_pixel_positions:
            messagebox.showerror(
                "Error",
                "No ChArUco board detected.\nMake sure the whole board is visible and try again."
            )
            return

        # Copy current data
        self.captured_corner_poses = dict(self.current_corner_poses)
        self.captured_pixel_positions = dict(self.charuco_pixel_positions)
        self.captured_corner_ids = sorted(self.captured_corner_poses.keys())
        self.board_captured = True

        # Update status text
        self.board_status_var.set(
            f"✓ Board captured ({len(self.captured_corner_ids)} corners)"
        )
        if hasattr(self, "board_status_label"):
            self.board_status_label.configure(fg="green")

        # Enable and update dropdown values to the captured corners
        display_values = [f"C{cid}" for cid in self.captured_corner_ids]
        self.corner_id_combo["values"] = display_values
        self.corner_id_combo.configure(state="readonly")

        # Keep the current selection if possible, otherwise pick the first one
        if self.selected_corner_id is None and self.captured_corner_ids:
            self.selected_corner_id = self.captured_corner_ids[0]

        desired = f"C{self.selected_corner_id}" if self.selected_corner_id is not None else ""
        if self.corner_id_var.get() != desired:
            self.corner_id_var.set(desired)

    # ---------- RealSense ----------

    def _init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)
        config.enable_stream(rs.stream.depth, COLOR_WIDTH, COLOR_HEIGHT, rs.format.z16, COLOR_FPS)

        print("Starting RealSense pipeline...")
        profile = self.pipeline.start(config)
        print("Pipeline started.")

        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float64)
      

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
                self._process_frame(frame_bgr)
            except Exception as e:
                print("Capture loop error:", e)
                time.sleep(0.1)

    def _process_frame(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Basic marker detection
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        charuco_pixel_positions = {}
        visible_ids = []

        if ids is not None and len(ids) > 0:
            # Interpolate ChArUco corners
            result = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )

            charuco_corners = None
            charuco_ids = None

            if isinstance(result, tuple):
                if len(result) == 3:
                    _, charuco_corners, charuco_ids = result
                elif len(result) == 2:
                    charuco_corners, charuco_ids = result

            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_ids) > 0
            ):
                # Build correspondences for PnP
                obj_pts = []
                img_pts = []
                for i in range(len(charuco_ids)):
                    cid = int(charuco_ids[i][0])
                    px, py = charuco_corners[i][0]

                    img_pts.append([px, py])
                    obj_pts.append(self.board_corners_3d[cid])

                obj_pts = np.asarray(obj_pts, dtype=np.float64)
                img_pts = np.asarray(img_pts, dtype=np.float64)

                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R_board_cam, _ = cv2.Rodrigues(rvec)
                    t_board_cam = tvec.reshape(3)

                    # Compute 3D positions for ALL detected corners
                    for i in range(len(charuco_ids)):
                        cid = int(charuco_ids[i][0])
                        px, py = charuco_corners[i][0]

                        charuco_pixel_positions[cid] = (px, py)
                        visible_ids.append(cid)

                        X_board = self.board_corners_3d[cid]
                        X_cam = R_board_cam @ X_board + t_board_cam
                        self.current_corner_poses[cid] = X_cam

        self.visible_corner_ids = sorted(set(visible_ids))
        self.charuco_pixel_positions = charuco_pixel_positions

        if self.board_captured and self.captured_pixel_positions:
            pixel_positions = self.captured_pixel_positions
        else:
            pixel_positions = {}  # no drawing before capture

        # Draw only the selected corner 
        if (
            self.board_captured
            and self.selected_corner_id is not None
            and pixel_positions
            and self.selected_corner_id in pixel_positions
        ):
            px, py = pixel_positions[self.selected_corner_id]
            cv2.rectangle(
                frame_bgr,
                (int(px) - 4, int(py) - 4),
                (int(px) + 4, int(py) + 4),
                (0, 0, 255),
                thickness=1
            )
            cv2.putText(
                frame_bgr,
                f"C{self.selected_corner_id}",
                (int(px) + 5, int(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        self.latest_frame = frame_bgr

    def _update_image(self):
        if self.latest_frame is not None:
            # Convert to RGB for Tk
            frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape

            # Initialize center if needed
            self._ensure_center_initialized(frame_rgb)

            # Compute region size based on zoom (crop then upscale)
            zoom = max(self.zoom_scale, 1.0)
            region_w = int(w / zoom)
            region_h = int(h / zoom)

            region_w = max(1, min(region_w, w))
            region_h = max(1, min(region_h, h))

            half_w = region_w // 2
            half_h = region_h // 2

            cx = int(self.center_x)
            cy = int(self.center_y)

            cx = max(half_w, min(w - half_w - 1, cx))
            cy = max(half_h, min(h - half_h - 1, cy))
            self.center_x, self.center_y = cx, cy

            x1 = cx - half_w
            x2 = cx + half_w
            y1 = cy - half_h
            y2 = cy + half_h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame_rgb[y1:y2, x1:x2]

            img = Image.fromarray(crop)
            if (crop.shape[1], crop.shape[0]) != (w, h):
                img = img.resize((w, h), Image.BILINEAR)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update corner dropdown
            if self.board_captured and self.captured_corner_ids:
                new_ids = list(self.captured_corner_ids)
            else:
                new_ids = []

            display_values = [f"C{cid}" for cid in new_ids]
            current_values = list(self.corner_id_combo["values"])

            # Only touch combo values if they actually changed
            if display_values != current_values:
                self.corner_id_combo["values"] = display_values

            if not self.board_captured:
                # Before capture: clear selection
                self.selected_corner_id = None
                if self.corner_id_var.get() != "":
                    self.corner_id_var.set("")
            else:
                # After capture, auto-select the first captured corner if none selected yet
                if self.selected_corner_id is None and new_ids:
                    self.selected_corner_id = new_ids[0]

                desired = f"C{self.selected_corner_id}" if self.selected_corner_id is not None else ""
                if self.corner_id_var.get() != desired:
                    self.corner_id_var.set(desired)

        self.root.after(30, self._update_image)

    # ---------- SAMPLE CAPTURE & CALIBRATION ----------

    def on_capture_sample(self):
        # Require a captured board so samples are based only on frozen data
        if not self.board_captured:
            messagebox.showerror(
                "Error",
                "Please capture the board first.\n"
                "Click 'Capture Board' while the full ChArUco board is visible."
            )
            return

        # Corner must come from dropdown selection
        raw_id = self.corner_id_var.get().strip()
        if not raw_id:
            messagebox.showerror("Error", "Please select a ChArUco corner from the dropdown.")
            return

        # Accept "C6" or "6"
        if raw_id.upper().startswith("C"):
            raw_id = raw_id[1:]

        try:
            corner_id = int(raw_id)
        except ValueError:
            messagebox.showerror("Error", "Corner ID must be an integer (e.g. 6 or C6).")
            return

        try:
            x_mm = float(self.robot_x_entry.get())
            y_mm = float(self.robot_y_entry.get())
            z_mm = float(self.robot_z_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Robot X/Y/Z must be numeric (mm).")
            return

        # Use captured board pose only
        if corner_id in self.captured_corner_poses:
            cam_corner_mm = self.captured_corner_poses[corner_id].copy()
        else:
            messagebox.showerror(
                "Error",
                f"Corner C{corner_id} is not in the captured board.\n"
                "Recapture the board if you moved it."
            )
            return

        try:
            z_offset_mm = float(self.z_offset_var.get())
        except ValueError:
            messagebox.showerror("Error", "Z offset must be numeric (mm).")
            return

        # Apply constant base-Z correction
        z_mm_corrected = z_mm - z_offset_mm

        robot_point_mm = np.array([x_mm, y_mm, z_mm_corrected], dtype=np.float64)

        sample = {
            "corner_id": corner_id,
            "camera_corner_mm": cam_corner_mm,
            "robot_point_mm": robot_point_mm,
        }
        self.samples.append(sample)

        idx = len(self.samples)
        self.samples_list.insert(
            tk.END,
            f"{idx:02d}: C{corner_id:03d}, "
            f"cam(mm)=({cam_corner_mm[0]:.1f}, {cam_corner_mm[1]:.1f}, {cam_corner_mm[2]:.1f}), "
            f"robot(mm)=({x_mm:.1f}, {y_mm:.1f}, {z_mm_corrected:.1f})"
        )

        self.robot_x_entry.delete(0, tk.END)
        self.robot_y_entry.delete(0, tk.END)
        self.robot_z_entry.delete(0, tk.END)

    def on_calibrate(self):
        if len(self.samples) < 3:
            messagebox.showwarning("Not enough samples", "Need at least 3 samples to calibrate. (6 to 12 samples recommended)")
            return

        cam_pts_mm = np.array(
            [s["camera_corner_mm"] for s in self.samples],
            dtype=np.float64
        )
        robot_pts_mm = np.array(
            [s["robot_point_mm"] for s in self.samples],
            dtype=np.float64
        )

        # camera -> base: robot = R * camera + t
        R_cb, t_cb = estimate_rigid_transform_3d(cam_pts_mm, robot_pts_mm)
        q_cb = rotation_matrix_to_quaternion_xyzw(R_cb)

        # base -> camera
        R_bc, t_bc = invert_transform(R_cb, t_cb)
        q_bc = rotation_matrix_to_quaternion_xyzw(R_bc)

        self.calibration_result = {
            "camera_to_base": {
                "translation_mm": t_cb.tolist(),
                "rotation_xyzw": q_cb.tolist(),
            },
            "base_to_camera": {
                "translation_mm": t_bc.tolist(),
                "rotation_xyzw": q_bc.tolist(),
            },
        }

        self._display_result()
        self.save_button.config(state=tk.NORMAL)

    def _display_result(self):
        self.result_text.delete("1.0", tk.END)

        res = self.calibration_result
        t_cb = res["camera_to_base"]["translation_mm"]
        q_cb = res["camera_to_base"]["rotation_xyzw"]
        t_bc = res["base_to_camera"]["translation_mm"]
        q_bc = res["base_to_camera"]["rotation_xyzw"]

        self.result_text.insert(tk.END, "== Camera (color optical) -> Robot Base ==\n")
        self.result_text.insert(tk.END, f"translation (mm):\n  [{t_cb[0]:.6f}, {t_cb[1]:.6f}, {t_cb[2]:.6f}]\n")
        self.result_text.insert(
            tk.END,
            "rotation (x, y, z, w):\n"
            f"  [{q_cb[0]:.8f}, {q_cb[1]:.8f}, {q_cb[2]:.8f}, {q_cb[3]:.8f}]\n\n"
        )

        self.result_text.insert(tk.END, "JSON camera_to_base (translation in meters):\n")
        json_style = {
            "translation": [t_cb[0] / 1000.0, t_cb[1] / 1000.0, t_cb[2] / 1000.0],
            "rotation": q_cb,
        }
        self.result_text.insert(tk.END, json.dumps(json_style, indent=2))
        self.result_text.insert(tk.END, "\n\n")

        self.result_text.insert(tk.END, "== Robot Base -> Camera (inverse) ==\n")
        self.result_text.insert(tk.END, f"translation (mm):\n  [{t_bc[0]:.6f}, {t_bc[1]:.6f}, {t_bc[2]:.6f}]\n")
        self.result_text.insert(
            tk.END,
            "rotation (x, y, z, w):\n"
            f"  [{q_bc[0]:.8f}, {q_bc[1]:.8f}, {q_bc[2]:.8f}, {q_bc[3]:.8f}]\n"
        )

    def on_save_json(self):
        if not hasattr(self, "calibration_result"):
            messagebox.showerror("Error", "No calibration result to save.")
            return

        c2b = self.calibration_result["camera_to_base"]
        t_mm = c2b["translation_mm"]
        q = c2b["rotation_xyzw"]

        json_data = {
            "translation": [t_mm[0] / 1000.0, t_mm[1] / 1000.0, t_mm[2] / 1000.0],
            "rotation": q,
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=2)

        messagebox.showinfo("Saved", f"Calibration saved to:\n{file_path}")

    # ---------- CLEANUP ----------

    def on_close(self):
        self.running = False
        time.sleep(0.1)
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            print("Error stopping pipeline:", e)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandEyeCharucoGUI(root)
    root.mainloop()
