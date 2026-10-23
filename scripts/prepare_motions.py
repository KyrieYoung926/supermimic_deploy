"""Convert G1 LAFAN1 CSVs to deployment references without a training installation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from motion_convert import load_csv, load_g1_model, resample_motion, forward_kinematics, quat_rotate_inverse, G1_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw/lafan1/g1'))
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    args = parser.parse_args()
    cfg = json.loads((args.assets / 'config.json').read_text())
    assert cfg['joint_names'] == list(G1_JOINT_NAMES)
    paths = [args.input_dir / (name + '.csv') for name in cfg['clips']]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    model = load_g1_model(args.assets / 'robot/g1.xml')
    out = args.assets / 'motions'
    out.mkdir(exist_ok=True)
    for path in paths:
        motion = resample_motion(load_csv(path), 30., 50.)
        bodies = forward_kinematics(model, motion)
        root = bodies.body_names.index('pelvis')
        q = bodies.quat_w[:, root].astype(np.float32)
        pos = bodies.pos_w[:, root].astype(np.float32)
        vel = bodies.lin_vel_w[:, root].astype(np.float32)
        omega = bodies.ang_vel_w[:, root].astype(np.float32)
        joints = motion.joint_pos.astype(np.float32)
        dq = motion.joint_vel.astype(np.float32)
        gravity = quat_rotate_inverse(q, np.broadcast_to([0., 0., -1.], vel.shape))
        ang_b = quat_rotate_inverse(q, omega)
        w, x, y, z = q.T
        yaw = np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
        lin_h = Rotation.from_euler('z', yaw[:, None]).inv().apply(vel)
        goal = np.concatenate([joints - np.array(cfg['q0'], dtype=np.float32), dq, gravity, ang_b, lin_h, pos[:, 2:3]], axis=-1).astype(np.float32)
        assert goal.shape == (len(joints), 68) and np.isfinite(goal).all()
        np.savez_compressed(out / (path.stem + '.npz'), goal=goal, joint_pos=joints,
                            joint_vel=dq, root_pos=pos, root_quat=q, root_lin_vel=vel,
                            root_ang_vel_b=ang_b.astype(np.float32), fps=np.array([50]))
        print(f'{path.stem}: {len(joints)} frames', flush=True)
    print(f'Prepared {len(paths)} motions in {out}')


if __name__ == '__main__':
    main()
