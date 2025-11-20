import threading
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import cv2
from PIL import Image, ImageTk
import pyrealsense2 as rs


# ===================== CONFIG =====================

# ChArUco board
BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 5

# These should match your PDF generator
SQUARE_LENGTH_MM = 39.0   # full square size
MARKER_LENGTH_MM = 31.0   # inner marker

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# RealSense color stream config
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30

# Use only these 6 ChArUco corner IDs as targets
# Adjust this list if you want different corners
TARGET_CORNER_IDS = [0, 5, 11, 12, 18, 23]


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
    Return Nx3 array of ChArUco board corner positions in board frame (mm),
    compatible with different ArUco Python bindings.
    """
    pts = None

    if hasattr(board, "getChessboardCorners"):
        pts = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        pts = board.chessboardCorners
    elif hasattr(board, "getObjPoints"):
        pts = board.getObjPoints()
    else:
        raise RuntimeError(
            "CharucoBoard has no getChessboardCorners/chessboardCorners/getObjPoints"
        )

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
        self.root.title("Hand-Eye Calibration (RealSense + ChArUco)")

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
        self.zoom_scale = 1.0  # for Zoom + / Zoom -

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

        # Video
        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # Zoom controls
        zoom_frame = ttk.Frame(video_frame)
        zoom_frame.pack(pady=4)
        ttk.Button(zoom_frame, text="Zoom -", command=self.on_zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom +", command=self.on_zoom_in).pack(side=tk.LEFT, padx=2)

        # Controls
        ctrl = ttk.Frame(main)
        ctrl.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(ctrl, text="Robot input (base frame, mm)")
        input_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(input_frame, text="ChArUco Corner ID:").grid(row=0, column=0, sticky="e")
        self.corner_id_entry = ttk.Entry(input_frame, width=10)
        self.corner_id_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        # Show which corners we care about
        target_str = ", ".join(f"C{cid}" for cid in TARGET_CORNER_IDS)
        ttk.Label(input_frame, text=f"Target corners: {target_str}").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4)
        )

        ttk.Label(input_frame, text="Robot X (mm):").grid(row=2, column=0, sticky="e")
        self.robot_x_entry = ttk.Entry(input_frame, width=10)
        self.robot_x_entry.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(input_frame, text="Robot Y (mm):").grid(row=3, column=0, sticky="e")
        self.robot_y_entry = ttk.Entry(input_frame, width=10)
        self.robot_y_entry.grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(input_frame, text="Robot Z (mm):").grid(row=4, column=0, sticky="e")
        self.robot_z_entry = ttk.Entry(input_frame, width=10)
        self.robot_z_entry.grid(row=4, column=1, sticky="w", padx=4, pady=2)

        ttk.Button(input_frame, text="Capture Sample", command=self.on_capture_sample)\
            .grid(row=5, column=0, columnspan=2, pady=4)

        # Samples
        samples_frame = ttk.LabelFrame(ctrl, text="Samples")
        samples_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.samples_list = tk.Listbox(samples_frame, height=8)
        self.samples_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bottom controls
        bottom = ttk.Frame(ctrl)
        bottom.pack(fill=tk.X)

        self.calibrate_button = ttk.Button(bottom, text="Calibrate", command=self.on_calibrate)
        self.calibrate_button.pack(side=tk.LEFT, padx=(0, 4))

        self.save_button = ttk.Button(bottom, text="Save JSON", command=self.on_save_json, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 4))

        self.result_text = tk.Text(ctrl, height=12, wrap="word")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # Initialize the corner ID entry with the first target
        if TARGET_CORNER_IDS:
            self.corner_id_entry.insert(0, str(TARGET_CORNER_IDS[0]))

    # ---------- Zoom handlers ----------

    def on_zoom_in(self):
        self.zoom_scale = min(self.zoom_scale * 1.25, 3.0)

    def on_zoom_out(self):
        self.zoom_scale = max(self.zoom_scale / 1.25, 0.5)

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
        depth_stream = profile.get_stream(rs.stream.depth)

        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float64)

        print("Color intrinsics (camera_color_optical_frame):")
        print(f"  fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
        print(f"  ppx: {color_intrinsics.ppx}, ppy: {color_intrinsics.ppy}")
        print(f"  dist: {self.dist_coeffs}")

        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        print("Depth intrinsics:")
        print(f"  fx: {depth_intrinsics.fx}, fy: {depth_intrinsics.fy}")
        print(f"  ppx: {depth_intrinsics.ppx}, ppy: {depth_intrinsics.ppy}")
        print(f"  dist: {np.array(depth_intrinsics.coeffs, dtype=np.float64)}")

        print("\nNOTE: We are calibrating w.r.t COLOR optical frame.")

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

        # IMPORTANT: do NOT clear current_corner_poses here.
        # We want to remember previously detected 3D positions
        # as long as the camera and board do not move.
        if ids is None or len(ids) == 0:
            self.latest_frame = frame_bgr
            return

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
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < 1
        ):
            self.latest_frame = frame_bgr
            return

        # Build correspondences for PnP (use all detected corners for robustness)
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
        if not success:
            self.latest_frame = frame_bgr
            return

        R_board_cam, _ = cv2.Rodrigues(rvec)
        t_board_cam = tvec.reshape(3)

        # Compute 3D positions and draw only the target corners
        for i in range(len(charuco_ids)):
            cid = int(charuco_ids[i][0])
            if cid not in TARGET_CORNER_IDS:
                continue

            px, py = charuco_corners[i][0]

            # compute 3D point in camera frame
            X_board = self.board_corners_3d[cid]
            X_cam = R_board_cam @ X_board + t_board_cam
            # cache the last known pose for this corner id
            self.current_corner_poses[cid] = X_cam

            # draw a small RED box on the exact corner
            cv2.rectangle(
                frame_bgr,
                (int(px) - 4, int(py) - 4),
                (int(px) + 4, int(py) + 4),
                (0, 0, 255),
                thickness=1
            )

            # draw the corner ID above it
            cv2.putText(
                frame_bgr,
                f"C{cid}",
                (int(px) + 5, int(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        self.latest_frame = frame_bgr

    def _update_image(self):
        if self.latest_frame is not None:
            frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Apply zoom
            if self.zoom_scale != 1.0:
                w, h = img.size
                img = img.resize(
                    (int(w * self.zoom_scale), int(h * self.zoom_scale)),
                    Image.BILINEAR
                )

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self._update_image)

    # ---------- SAMPLE CAPTURE & CALIBRATION ----------

    def on_capture_sample(self):
        if len(self.samples) >= len(TARGET_CORNER_IDS):
            messagebox.showinfo(
                "Samples complete",
                f"You already captured {len(TARGET_CORNER_IDS)} samples."
            )
            return

        # Corner ID can be "6" or "C6"
        raw_id = self.corner_id_entry.get().strip()
        if raw_id.upper().startswith("C"):
            raw_id = raw_id[1:]

        try:
            corner_id = int(raw_id)
        except ValueError:
            messagebox.showerror("Error", "Corner ID must be an integer (e.g. 6 or C6).")
            return

        if corner_id not in TARGET_CORNER_IDS:
            messagebox.showerror(
                "Error",
                f"Corner C{corner_id} is not in target list.\n"
                f"Use one of: {', '.join('C'+str(c) for c in TARGET_CORNER_IDS)}"
            )
            return

        try:
            x_mm = float(self.robot_x_entry.get())
            y_mm = float(self.robot_y_entry.get())
            z_mm = float(self.robot_z_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Robot X/Y/Z must be numeric (mm).")
            return

        if corner_id not in self.current_corner_poses:
            messagebox.showerror(
                "Error",
                f"Corner C{corner_id} has never been detected yet.\n"
                "Make sure it is visible at least once so the camera can estimate its 3D position."
            )
            return

        cam_corner_mm = self.current_corner_poses[corner_id].copy()
        robot_point_mm = np.array([x_mm, y_mm, z_mm], dtype=np.float64)

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
            f"robot(mm)=({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f})"
        )

        self.robot_x_entry.delete(0, tk.END)
        self.robot_y_entry.delete(0, tk.END)
        self.robot_z_entry.delete(0, tk.END)

        # Auto-advance to the next target corner that hasn't been used yet
        used_ids = {s["corner_id"] for s in self.samples}
        next_id = None
        for cid in TARGET_CORNER_IDS:
            if cid not in used_ids:
                next_id = cid
                break

        self.corner_id_entry.delete(0, tk.END)
        if next_id is not None:
            self.corner_id_entry.insert(0, str(next_id))
        else:
            # All done
            self.corner_id_entry.insert(0, "")

    def on_calibrate(self):
        if len(self.samples) < 3:
            messagebox.showwarning("Not enough samples", "Need at least 3, 6–12 is better.")
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
