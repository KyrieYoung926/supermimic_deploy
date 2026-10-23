"""G1 29-DoF HG/PR deployment, following Unitree SDK2 low-level example."""

import argparse
import struct
import threading
import time
from pathlib import Path

import numpy as np
from controller import B, Controller


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interface", required=True, help="Ethernet interface connected to G1")
    p.add_argument(
        "--robot",
        choices=["g1-29dof"],
        required=True,
        help="Requires unlocked 29-DoF G1, pelvis IMU",
    )
    p.add_argument("--assets", type=Path, default=Path(__file__).parent / "assets")
    p.add_argument("--motion", default="walk1_subject1")
    p.add_argument("--align-seconds", type=float, default=3.0)
    p.add_argument("--skip-start-pose-check", action="store_true",
                   help="Skip joint/gravity alignment checks on A; Start alignment is still required")
    args = p.parse_args()
    # Validate assets and warm up inference BEFORE connecting to robot.
    policy = Controller(args.assets, args.motion, args.align_seconds,
                        skip_start_pose_check=args.skip_start_pose_check,
                        allow_motion_switch=True)
    if args.skip_start_pose_check:
        print("Start pose checks are disabled. Extra dexterous-hand mass and inertia are not modeled by this policy.", flush=True)
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, args.interface)
    lock = threading.Lock()
    fault = threading.Event()
    stop = threading.Event()
    shared = {"snapshot": None, "tick": None, "target": None, "target_time": 0.0, "mode": None}
    crc = CRC()

    def receive(msg):
        if CRC().Crc(msg) != msg.crc:
            return
        q = np.array([m.q for m in msg.motor_state[:29]], np.float32)
        dq = np.array([m.dq for m in msg.motor_state[:29]], np.float32)
        quat = np.array(msg.imu_state.quaternion, np.float32)
        omega = np.array(msg.imu_state.gyroscope, np.float32)
        keys = struct.unpack("<H", bytes(msg.wireless_remote[2:4]))[0]
        with lock:
            if shared["tick"] == msg.tick:
                return
            shared["tick"] = msg.tick
            if shared["mode"] is not None and shared["mode"] != msg.mode_machine:
                fault.set()
            shared["snapshot"] = (time.monotonic(), q, dq, quat, omega, keys, msg.mode_machine)
            if keys & B:
                fault.set()
                shared["target"] = None

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(receive, 10)
    deadline = time.monotonic() + 10
    while shared["snapshot"] is None:
        if time.monotonic() > deadline:
            raise RuntimeError("No valid LowState within 10 seconds")
        time.sleep(0.02)
    shared["mode"] = shared["snapshot"][-1]
    switch = MotionSwitcherClient()
    switch.SetTimeout(5.0)
    switch.Init()
    for attempt in range(6):
        code, result = switch.CheckMode()
        if code != 0 or not isinstance(result, dict):
            raise RuntimeError(f"CheckMode failed: {code}")
        if not result.get("name"):
            break
        if attempt == 5:
            raise RuntimeError("Existing motion service still active")
        release_code, _ = switch.ReleaseMode()
        if release_code != 0:
            raise RuntimeError("ReleaseMode failed")
        time.sleep(1)
    else:
        raise RuntimeError("Existing motion service still active")
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = shared["mode"]
    for i, m in enumerate(cmd.motor_cmd):
        m.mode = 1 if i < 29 else 0
        m.q = m.dq = m.tau = m.kp = m.kd = 0.0

    def writer():
        deadline = time.monotonic()
        try:
            while not stop.is_set():
                now = time.monotonic()
                with lock:
                    snap = shared["snapshot"]
                    target = shared["target"]
                    stamp = shared["target_time"]
                stale = now - snap[0] > 0.1 or snap[-1] != shared["mode"]
                invalid = not all(np.isfinite(x).all() for x in snap[1:5])
                if stale or invalid or (target is not None and now - stamp > 0.1):
                    fault.set()
                damping = fault.is_set() or target is None or bool(snap[5] & B)
                if not damping:
                    target = policy.bounded_target(target, snap[1], snap[2])
                    if not np.isfinite(target).all():
                        damping = True
                        fault.set()
                for i in range(29):
                    m = cmd.motor_cmd[i]
                    m.q = 0.0 if damping else float(target[i])
                    m.dq = 0.0
                    m.tau = 0.0
                    m.kp = 0.0 if damping else float(policy.kp[i])
                    m.kd = 2.0 if damping else float(policy.kd[i])
                cmd.crc = crc.Crc(cmd)
                if not pub.Write(cmd):
                    fault.set()
                deadline += 0.002
                time.sleep(max(0, deadline - time.monotonic()))
                if time.monotonic() - deadline > 0.01:
                    deadline = time.monotonic()
        except Exception as e:  # noqa: BLE001 -- stop command thread on any transport failure
            fault.set()
            stop.set()
            print(f"Command writer failed: {e}", flush=True)

    thread = threading.Thread(target=writer, name="dds-command-500hz", daemon=True)
    thread.start()
    print("DAMPING. Start: align; A: run; B: damp; release B, then X/Y: previous/next motion.", flush=True)
    old_state = None
    deadline = time.monotonic()
    try:
        while not stop.is_set():
            now = time.monotonic()
            with lock:
                snap = shared["snapshot"]
            if fault.is_set():
                policy.damp("B / sensor or inference watchdog")
                # A held Start must not re-arm the controller after a fault.
                policy.last_keys = snap[5]
                with lock:
                    shared["target"] = None
                if (
                    now - snap[0] <= 0.1
                    and not snap[5] & B
                    and snap[-1] == shared["mode"]
                    and all(np.isfinite(x).all() for x in snap[1:5])
                ):
                    fault.clear()
                target = None
            elif now - snap[0] > 0.1 or snap[-1] != shared["mode"]:
                policy.damp("stale state or machine mode changed")
                target = None
            else:
                target = policy.tick(now, *snap[1:6])
            if time.monotonic() - now > 0.1:
                policy.damp("inference exceeded 100ms")
                target = None
                fault.set()
            with lock:
                shared["target"] = None if fault.is_set() else target
                shared["target_time"] = time.monotonic()
            status = (policy.state, policy.reason, policy.motion)
            if status != old_state:
                print(f"{policy.state.value}: {policy.reason}; motion={policy.motion}; frame={policy.frame}", flush=True)
                old_state = status
            deadline += 0.02
            time.sleep(max(0, deadline - time.monotonic()))
            if time.monotonic() - deadline > 0.02:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        print("Exiting in damping", flush=True)
    finally:
        fault.set()
        with lock:
            shared["target"] = None
        # Publish damping repeatedly before closing the sender.
        time.sleep(0.5)
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
