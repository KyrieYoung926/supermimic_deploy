"""Hardware-independent policy/state machine. Arrays use training joint order."""

import enum
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

START, A, B = 1 << 2, 1 << 8, 1 << 9
X, Y = 1 << 10, 1 << 11


def gravity(quat):
    q = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 0.5 or norm > 1.5:
        raise ValueError("Invalid IMU quaternion")
    w, x, y, z = q / norm
    return np.array(
        [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1], dtype=np.float32
    )


def flatten_history(history):
    return np.concatenate(
        [
            history[:, lo:hi].reshape(-1)
            for lo, hi in ((0, 29), (29, 58), (58, 61), (61, 64), (64, 93))
        ]
    )[None].astype(np.float32)


class State(enum.Enum):
    DAMPING = "damping"
    ALIGN = "moving to first frame"
    READY = "holding first frame"
    POLICY = "policy"


class Controller:
    def __init__(self, assets, motion="walk1_subject1", align_seconds=3.0,
                 skip_start_pose_check=False, allow_motion_switch=False):
        assets = Path(assets)
        self.cfg = json.loads((assets / "config.json").read_text())
        if self.cfg["anchor_obs"] or self.cfg["motor_ids"] != list(range(29)):
            raise ValueError("Unsupported bundle")
        if (
            hashlib.sha256((assets / "policy.onnx").read_bytes()).hexdigest()
            != self.cfg["onnx_sha256"]
        ):
            raise ValueError("ONNX hash mismatch")
        for key in ("q0", "scale", "kp", "kd", "effort"):
            setattr(self, key, np.array(self.cfg[key], dtype=np.float32))
        if align_seconds < 2:
            raise ValueError("Alignment must be >= 2 seconds")
        self.align_seconds = align_seconds
        self.skip_start_pose_check = skip_start_pose_check
        self.allow_motion_switch = allow_motion_switch
        self.motion_dir = assets / "motions"
        self.motions = tuple(self.cfg["clips"])
        self.select_motion(motion)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(assets / "policy.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.session.run(
            None,
            {
                "obs_history": np.zeros((1, 930), np.float32),
                "goal": self.goals[:10].reshape(1, 680),
            },
        )
        self.state = State.DAMPING
        self.history = np.zeros((10, 93), np.float32)
        self.previous_action = np.zeros(29, np.float32)
        self.last_keys = 0
        self.frame = 0
        self.reason = "startup"

    def select_motion(self, motion):
        """Select a clip; the caller must align and confirm before policy resumes."""
        if motion not in self.motions:
            raise ValueError(f"Unknown motion: {motion}")
        with np.load(self.motion_dir / f"{motion}.npz") as d:
            goals = d["goal"]
            positions = d["joint_pos"]
            ref_quat = d["root_quat"]
        if (
            goals.shape != (len(positions), 68)
            or positions.shape != (len(goals), 29)
            or ref_quat.shape != (len(goals), 4)
            or len(goals) == 0
            or not all(np.isfinite(x).all() for x in (goals, positions, ref_quat))
        ):
            raise ValueError(f"Invalid motion: {motion}")
        self.motion = motion
        self.goals, self.positions, self.ref_quat = goals, positions, ref_quat
        if hasattr(self, "state"):
            self.damp(f"selected {motion}; press Start to align")
            self.frame = 0
            self.history.fill(0)

    def damp(self, reason):
        self.state = State.DAMPING
        self.reason = reason
        self.previous_action.fill(0)

    def tick(self, now, q, dq, quat, omega, keys):
        q, dq, omega = [np.asarray(x, np.float32) for x in (q, dq, omega)]
        if (
            q.shape != (29,)
            or dq.shape != (29,)
            or omega.shape != (3,)
            or not all(np.isfinite(x).all() for x in (q, dq, omega))
        ):
            self.damp("invalid sensor data")
            return None
        rising = keys & ~self.last_keys
        self.last_keys = keys
        if keys & B:
            self.damp("B pressed")
            return None
        if self.allow_motion_switch and rising & (X | Y):
            if self.state == State.DAMPING:
                if bool(rising & X) != bool(rising & Y):
                    direction = -1 if rising & X else 1
                    index = self.motions.index(self.motion)
                    self.select_motion(self.motions[(index + direction) % len(self.motions)])
                return None
            self.reason = "Motion switch rejected: press B to damp, release B, then press X/Y"
        if rising & A and self.state != State.READY:
            self.reason = "A rejected: press Start, wait for alignment, then release and press A"
        try:
            grav = gravity(quat)
        except ValueError:
            self.damp("invalid IMU")
            return None
        if rising & START and self.state in (State.DAMPING, State.READY):
            self.start_q = q.copy()
            self.start_time = now
            self.state = State.ALIGN
            self.reason = "Aligning to the first frame; wait for completion before pressing A"
            self.frame = 0
            self.previous_action.fill(0)
            self.history.fill(0)
        if self.state == State.ALIGN:
            t = np.clip((now - self.start_time) / self.align_seconds, 0, 1)
            blend = t * t * (3 - 2 * t)
            target = self.start_q * (1 - blend) + self.positions[0] * blend
            if t >= 1:
                self.state = State.READY
                self.reason = "Holding the first frame; release and press A to run"
            return target
        if self.state == State.READY:
            errors = np.abs(q - self.positions[0])
            worst = int(np.argmax(errors))
            gravity_error = float(np.linalg.norm(grav - gravity(self.ref_quat[0])))
            if (
                rising & A
                and (self.skip_start_pose_check or (
                    errors[worst] < 0.15 and gravity_error < 0.35
                ))
            ):
                self.state = State.POLICY
                self.reason = "Policy running (start pose checks skipped)" if self.skip_start_pose_check else "Policy running"
                obs = np.concatenate((q - self.q0, dq, omega, grav, self.previous_action))
                self.history[:] = obs
            else:
                if rising & A:
                    self.reason = (
                        f"A rejected: largest joint error {self.cfg['joint_names'][worst]}="
                        f"{errors[worst]:.3f} rad (required <0.150), "
                        f"gravity direction error={gravity_error:.3f} (required <0.350); release and press A again"
                    )
                return self.positions[0].copy()
        if self.state == State.POLICY:
            if self.frame >= len(self.goals):
                self.damp("motion complete")
                return None
            self.history[:-1] = self.history[1:]
            self.history[-1] = np.concatenate((q - self.q0, dq, omega, grav, self.previous_action))
            idx = np.minimum(np.arange(self.frame, self.frame + 10), len(self.goals) - 1)
            action = self.session.run(
                None,
                {
                    "obs_history": flatten_history(self.history),
                    "goal": self.goals[idx].reshape(1, 680),
                },
            )[0].reshape(29)
            if not np.isfinite(action).all():
                self.damp("nonfinite action")
                return None
            self.previous_action = action.copy()
            self.frame += 1
            return self.q0 + self.scale * action
        return None

    def bounded_target(self, target, q, dq):
        # Low-gain setpoints are torque proxies; saturate PD torque, not q_sp at physical joint limits.
        torque = np.clip(self.kp * (target - q) - self.kd * dq, -self.effort, self.effort)
        return np.asarray(q) + (torque + self.kd * np.asarray(dq)) / self.kp
