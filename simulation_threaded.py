"""
Threaded Simulation Wrapper - Non-blocking PyBullet control
Runs in DIRECT (headless) mode; frames rendered via get_camera_image()
for embedding inside the Qt window.
"""
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
from threading import Thread, Lock
import time
from kinematics import forward_kinematics, HOME_CONFIG, JOINT_LIMITS

# Prefer hardware renderer – falls back automatically on no-GPU systems
_RENDERER = p.ER_BULLET_HARDWARE_OPENGL

# Hard cap on render size to keep frame time low on high-DPI displays
_MAX_RENDER_W = 1600
_MAX_RENDER_H = 900



class ThreadedRobotSimulation:
    def __init__(self):
        self.lock = Lock()
        self.running = False
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = []

        # State
        self.current_q = HOME_CONFIG.copy()
        self.target_q = HOME_CONFIG.copy()
        self.is_moving = False
        self.stop_requested = False

        # Trajectory
        self.trajectory = []
        self.traj_index = 0

        # Camera orbit state
        self._cam_yaw      = 42.0
        self._cam_pitch    = -28.0
        self._cam_distance = 1.55
        self._cam_target   = [0.20, 0.0, 0.42]


        # Pre-rendered frame buffer (background thread fills this)
        self._frame_lock   = Lock()
        self._latest_frame = None

        # Desired render resolution — updated by ViewportLabel.resizeEvent()
        self._req_w = 640
        self._req_h = 360


    def start(self):
        """Initialize PyBullet in DIRECT mode and start simulation thread."""
        self.physics_client = p.connect(p.DIRECT)   # headless – no OS window
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240.0)

        # Load scene
        self.plane_id = p.loadURDF("plane.urdf")
        urdf_path = Path(__file__).parent / "assets" / "my_robot.urdf"
        self.robot_id = p.loadURDF(
            str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True)

        # Collect revolute joints
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        self.set_joint_angles(HOME_CONFIG)

        self.running = True

        # Physics simulation thread (240 Hz)
        self.thread = Thread(target=self._simulation_loop,
                             name="PhysicsThread", daemon=True)
        self.thread.start()

        # Viewport render thread (25 fps)
        self._render_thread = Thread(target=self._render_loop,
                                     name="RenderThread", daemon=True)
        self._render_thread.start()

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _simulation_loop(self):
        """Background thread – physics at 240 Hz."""
        while self.running:
            with self.lock:
                if self.is_moving and len(self.trajectory) > 0:
                    if self.stop_requested:
                        self.is_moving = False
                        self.stop_requested = False
                        self.trajectory = []
                        self.traj_index = 0
                    elif self.traj_index < len(self.trajectory):
                        q = self.trajectory[self.traj_index]
                        self._set_joints_internal(q)
                        self.traj_index += 1
                    else:
                        self.is_moving = False
                        self.trajectory = []
                        self.traj_index = 0

                self.current_q = self._get_joints_internal()

            p.stepSimulation()
            time.sleep(1 / 240.0)

    # ------------------------------------------------------------------
    # Internal helpers (call with lock held)
    # ------------------------------------------------------------------

    def _set_joints_internal(self, q):
        for idx, angle in zip(self.joint_indices[:6], q):
            p.resetJointState(self.robot_id, idx, angle)

    def _get_joints_internal(self):
        angles = []
        for idx in self.joint_indices[:6]:
            state = p.getJointState(self.robot_id, idx)
            angles.append(state[0])
        return np.array(angles)

    # ------------------------------------------------------------------
    # Public API – joints
    # ------------------------------------------------------------------

    def set_joint_angles(self, q):
        with self.lock:
            self._set_joints_internal(q)
            self.current_q = q.copy()

    def get_joint_angles(self):
        with self.lock:
            return self.current_q.copy()

    def get_end_effector_pose(self):
        with self.lock:
            link_state = p.getLinkState(
                self.robot_id, self.joint_indices[-1] + 1)
            pos = np.array(link_state[0])
            orn_quat = link_state[1]
            return pos, orn_quat

    def move_to_pose_smooth(self, target_pos, target_orn_quat, steps=50):
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.joint_indices[-1] + 1,
            target_pos.tolist(),
            target_orn_quat,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        if not solution:
            return False, "IK failed"

        target_q = np.array(solution[:6])
        if not np.all(
            (target_q >= JOINT_LIMITS[:, 0]) & (target_q <= JOINT_LIMITS[:, 1])
        ):
            return False, "Joint limits exceeded"

        with self.lock:
            current = self.current_q.copy()
            self.trajectory = [
                current + (target_q - current) * t
                for t in np.linspace(0, 1, steps)
            ]
            self.traj_index = 0
            self.is_moving = True
            self.stop_requested = False
            self.target_q = target_q

        return True, "Moving"

    def stop_motion(self):
        with self.lock:
            self.stop_requested = True

    def is_motion_active(self):
        with self.lock:
            return self.is_moving

    def show_target_sphere(self, position, color=None):
        if color is None:
            color = [1, 0, 0, 0.5]
        with self.lock:
            if hasattr(self, "target_sphere_id"):
                try:
                    p.removeBody(self.target_sphere_id)
                except Exception:
                    pass
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.02, rgbaColor=color)
            self.target_sphere_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
            )

    # ------------------------------------------------------------------
    # Background viewport renderer  (25 fps, never blocks UI)
    # ------------------------------------------------------------------

    # --– public: called from Qt main thread on viewport resize ------------------
    def set_render_size(self, w: int, h: int):
        """Thread-safe: update render resolution to match viewport physical pixels."""
        w = max(64, int(w))
        h = max(36, int(h))
        if w > _MAX_RENDER_W or h > _MAX_RENDER_H:
            sx = _MAX_RENDER_W / float(w)
            sy = _MAX_RENDER_H / float(h)
            s  = min(sx, sy)
            w  = max(64, int(w * s))
            h  = max(36, int(h * s))
        self._req_w = w
        self._req_h = h

    def _render_loop(self):
        """Pre-render frames into _latest_frame at ~30 fps at native resolution."""
        while self.running:
            w, h = self._req_w, self._req_h      # always native physical res
            frame = self.get_camera_image(w, h)
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(0.033)   # 30 fps cap


    def get_latest_frame(self):
        """
        Return the most recent pre-rendered frame (never blocks).
        Returns None until the first frame is ready.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    # ------------------------------------------------------------------
    # Camera orbit control (thread-safe)
    # ------------------------------------------------------------------

    def orbit_camera(self, dyaw: float, dpitch: float):
        """Adjust camera orbit angles (degrees)."""
        self._cam_yaw   += dyaw
        self._cam_pitch  = float(np.clip(self._cam_pitch + dpitch, -89, 0))

    def zoom_camera(self, delta: float):
        self._cam_distance = float(
            np.clip(self._cam_distance + delta, 0.3, 4.0))

    # ------------------------------------------------------------------
    # Embedded viewport rendering
    # ------------------------------------------------------------------

    def get_camera_image(self, width: int = 640, height: int = 360):
        """
        Render one frame. Uses hardware OpenGL when available,
        falls back to TINY_RENDERER automatically.
        Returns (H, W, 3) uint8 RGB array, or None on error.
        """
        try:
            view_mat = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._cam_target,
                distance=self._cam_distance,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2,
            )
            proj_mat = p.computeProjectionMatrixFOV(
                fov=55,
                aspect=width / max(height, 1),
                nearVal=0.01,
                farVal=12.0,
            )
            # Calibrated 3-point lighting — matches native PyBullet GUI look.
            # Lower ambient prevents washed-out look; strong diffuse gives depth.
            _, _, rgb_px, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                lightDirection=[1.2, 0.8, 2.0],  # key from top-front-right
                lightColor=[1.0, 1.0, 1.0],
                lightDistance=2.5,
                shadow=1,
                lightAmbientCoeff=0.28,    # low: no washed-out haze
                lightDiffuseCoeff=0.90,    # strong: crisp shading
                lightSpecularCoeff=0.20,   # modest: subtle highlights
                renderer=_RENDERER,
            )
            arr = np.array(rgb_px, dtype=np.uint8).reshape(height, width, 4)
            return arr[:, :, :3]    # drop alpha → (H, W, 3) RGB
        except Exception:
            # Hardware OGL unavailable — fall back silently to software
            try:
                view_mat = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=self._cam_target,
                    distance=self._cam_distance,
                    yaw=self._cam_yaw,
                    pitch=self._cam_pitch,
                    roll=0, upAxisIndex=2)
                proj_mat = p.computeProjectionMatrixFOV(
                    fov=55, aspect=width/max(height,1),
                    nearVal=0.01, farVal=12.0)
                _, _, rgb_px, _, _ = p.getCameraImage(
                    width=width, height=height,
                    viewMatrix=view_mat, projectionMatrix=proj_mat,
                    renderer=p.ER_TINY_RENDERER)
                arr = np.array(rgb_px, dtype=np.uint8).reshape(height, width, 4)
                return arr[:, :, :3]
            except Exception:
                return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self):
        self.running = False
        if hasattr(self, "thread") and self.thread:
            self.thread.join(timeout=1.0)
        if hasattr(self, "_render_thread") and self._render_thread:
            self._render_thread.join(timeout=0.5)
        try:
            p.disconnect()
        except Exception:
            pass
