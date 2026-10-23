"""Offline checks: never initialize a real DDS participant or send commands."""

import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

import deploy_real
from controller import A, B, START, X, Y, Controller, State, flatten_history, gravity
from unitree_sdk2py.core import channel
from unitree_sdk2py.comm.motion_switcher import motion_switcher_client
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC


class PipelineTests(unittest.TestCase):
    def run_main(self, switch, thread_class, policy=None, publisher=None):
        msg = unitree_hg_msg_dds__LowState_()
        msg.imu_state.quaternion = [1., 0., 0., 0.]
        msg.wireless_remote[2] = START
        msg.crc = CRC().Crc(msg)
        subscriber = Mock()
        subscriber.Init.side_effect = lambda callback, _: callback(msg)
        policy = policy or Mock()
        policy.tick.side_effect = KeyboardInterrupt
        with (
            patch.object(sys, 'argv', ['deploy_real.py', '--interface', 'offline', '--robot', 'g1-29dof']),
            patch.object(deploy_real, 'Controller', return_value=policy),
            patch.object(channel, 'ChannelFactoryInitialize') as initialize,
            patch.object(channel, 'ChannelSubscriber', return_value=subscriber),
            patch.object(channel, 'ChannelPublisher', return_value=publisher or Mock()),
            patch.object(motion_switcher_client, 'MotionSwitcherClient', return_value=switch),
            patch.object(deploy_real.threading, 'Thread', thread_class),
            patch.object(deploy_real.time, 'sleep'),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            deploy_real.main()
            initialize.assert_called_once_with(0, 'offline')
        return policy

    def test_release_mode_tuple_and_final_recheck(self):
        switch = Mock()
        switch.CheckMode.side_effect = [(0, {'name': 'sport'})] * 5 + [(0, {'name': ''})]
        switch.ReleaseMode.return_value = (0, None)
        self.run_main(switch, Mock())
        self.assertEqual(switch.ReleaseMode.call_count, 5)
        self.assertEqual(switch.CheckMode.call_count, 6)

    def test_release_failure_aborts(self):
        switch = Mock()
        switch.CheckMode.return_value = (0, {'name': 'sport'})
        switch.ReleaseMode.return_value = (123, None)
        with self.assertRaisesRegex(RuntimeError, 'ReleaseMode failed'):
            self.run_main(switch, Mock())

    def test_fault_consumes_held_start(self):
        class FaultThread:
            def __init__(self, target, **kwargs):
                self.cells = dict(zip(target.__code__.co_freevars, (c.cell_contents for c in target.__closure__)))

            def start(self):
                self.cells['fault'].set()

            def join(self, **kwargs):
                pass

        switch = Mock()
        switch.CheckMode.return_value = (0, {'name': ''})
        policy = self.run_main(switch, FaultThread)
        self.assertEqual(policy.last_keys, START)
        policy.damp.assert_called()

    def test_failed_write_latches_damping(self):
        captured = []
        publisher = Mock()

        class WriterThread:
            def __init__(self, target, **kwargs):
                self.target = target
                self.cells = dict(zip(target.__code__.co_freevars, (c.cell_contents for c in target.__closure__)))

            def start(self):
                self.cells['shared']['target'] = np.zeros(29)
                self.cells['shared']['target_time'] = deploy_real.time.monotonic()

                def write(cmd):
                    captured.append((cmd.motor_cmd[0].kp, cmd.motor_cmd[0].kd))
                    assert cmd.crc == CRC().Crc(cmd)
                    if len(captured) == 1:
                        return False
                    self.cells['stop'].set()
                    return True

                publisher.Write.side_effect = write
                self.target()

            def join(self, **kwargs):
                pass

        switch = Mock()
        switch.CheckMode.return_value = (0, {'name': ''})
        policy = Mock(kp=np.ones(29)*40, kd=np.ones(29), effort=np.ones(29)*50)
        policy.bounded_target.return_value = np.zeros(29)
        self.run_main(switch, WriterThread, policy, publisher)
        self.assertEqual(captured, [(40., 1.), (0., 2.)])

    def test_writer_watchdogs(self):
        for scenario in ('stale_state', 'stale_target', 'mode_change', 'invalid_state', 'B'):
            with self.subTest(scenario=scenario):
                captured = []
                publisher = Mock()

                class WriterThread:
                    def __init__(self, target, **kwargs):
                        self.target = target
                        self.cells = dict(zip(target.__code__.co_freevars, (c.cell_contents for c in target.__closure__)))

                    def start(self):
                        shared = self.cells['shared']
                        shared['target'] = np.zeros(29)
                        shared['target_time'] = deploy_real.time.monotonic()
                        snap = list(shared['snapshot'])
                        if scenario == 'stale_state':
                            snap[0] -= 1
                        elif scenario == 'stale_target':
                            shared['target_time'] -= 1
                        elif scenario == 'mode_change':
                            snap[-1] += 1
                        elif scenario == 'invalid_state':
                            snap[1][0] = np.nan
                        else:
                            snap[5] = B
                        shared['snapshot'] = tuple(snap)

                        def write(cmd):
                            captured.extend((m.kp, m.kd) for m in cmd.motor_cmd[:29])
                            self.cells['stop'].set()
                            return True

                        publisher.Write.side_effect = write
                        self.target()

                    def join(self, **kwargs):
                        pass

                switch = Mock()
                switch.CheckMode.return_value = (0, {'name': ''})
                self.run_main(switch, WriterThread, publisher=publisher)
                self.assertEqual(captured, [(0., 2.)] * 29)

    def test_start_pose_check_and_explicit_bypass(self):
        for bypass in (False, True):
            with self.subTest(bypass=bypass):
                p = Controller('assets', skip_start_pose_check=bypass)
                q = p.positions[0].copy()
                q[20] += .2
                zero, omega = np.zeros(29), np.zeros(3)
                quat = np.array([.70710678, .70710678, 0, 0])
                # Even bypass requires Start and completion of alignment.
                self.assertIsNone(p.tick(0, q, zero, quat, omega, A))
                self.assertEqual(p.state, State.DAMPING)
                p.tick(.02, q, zero, quat, omega, START)
                p.tick(4, q, zero, quat, omega, 0)
                p.tick(4.02, q, zero, quat, omega, A)
                self.assertEqual(p.state, State.POLICY if bypass else State.READY)
                if not bypass:
                    self.assertIn('left_wrist_pitch_joint', p.reason)
                p.tick(4.04, q, zero, quat, omega, B)
                self.assertEqual(p.state, State.DAMPING)

    def test_motion_switch_only_in_damping(self):
        p = Controller('assets', allow_motion_switch=True)
        q, dq, quat, omega = p.positions[0].copy(), np.zeros(29), p.ref_quat[0], np.zeros(3)
        original = p.motion
        p.previous_action.fill(1)
        p.history.fill(1)
        self.assertIsNone(p.tick(0, q, dq, quat, omega, Y | START | A))
        selected = p.motion
        self.assertNotEqual(selected, original)
        self.assertEqual(p.state, State.DAMPING)
        self.assertEqual(p.frame, 0)
        self.assertFalse(p.history.any())
        self.assertFalse(p.previous_action.any())
        p.tick(.02, q, dq, quat, omega, Y)
        self.assertEqual(p.motion, selected)  # no repeat while held
        p.tick(.04, q, dq, quat, omega, 0)
        p.tick(.06, q, dq, quat, omega, X)
        self.assertEqual(p.motion, original)
        for state in (State.READY, State.POLICY):
            p.state = state
            p.last_keys = 0
            p.tick(.08, q, dq, quat, omega, Y)
            self.assertEqual(p.motion, original)
        p.tick(.1, q, dq, quat, omega, B | X)
        self.assertEqual(p.state, State.DAMPING)
        self.assertEqual(p.motion, original)  # B has priority
        p.tick(.12, q, dq, quat, omega, 0)
        p.tick(.14, q, dq, quat, omega, X | Y)
        self.assertEqual(p.motion, original)

    def test_assets_state_machine_and_torque(self):
        p = Controller('assets')
        for motion in p.motions:
            p.select_motion(motion)
            self.assertGreaterEqual(len(p.goals), 10)
        p.select_motion('walk1_subject1')
        q = p.positions[0].copy()
        quat = p.ref_quat[0]
        zero = np.zeros(29)
        omega = np.zeros(3)
        self.assertIsNone(p.tick(0, q, zero, quat, omega, A))
        p.tick(.02, q, zero, quat, omega, START)
        self.assertEqual(p.state, State.ALIGN)
        p.tick(4, q, zero, quat, omega, 0)
        self.assertEqual(p.state, State.READY)
        target = p.tick(4.02, q, zero, quat, omega, A)
        self.assertEqual(p.state, State.POLICY)
        self.assertTrue(np.isfinite(target).all())
        dq = np.linspace(-30, 30, 29)
        bounded = p.bounded_target(np.ones(29)*1000, q, dq)
        torque = p.kp * (bounded-q) - p.kd*dq
        self.assertTrue(np.all(np.abs(torque) <= p.effort + 1e-4))
        self.assertIsNone(p.tick(4.04, q, zero, quat, omega, B | START | A))
        self.assertEqual(p.state, State.DAMPING)
        self.assertIsNone(p.tick(4.06, q, zero, np.zeros(4), omega, START))
        p.state = State.POLICY
        p.frame = len(p.goals)
        self.assertIsNone(p.tick(5, q, zero, quat, omega, 0))
        self.assertEqual(p.reason, 'motion complete')
        history = np.arange(930).reshape(10, 93)
        expected = np.concatenate([history[:, a:b].ravel() for a,b in [(0,29),(29,58),(58,61),(61,64),(64,93)]])
        np.testing.assert_array_equal(flatten_history(history)[0], expected)
        np.testing.assert_allclose(gravity([1,0,0,0]), [0,0,-1])


if __name__ == '__main__':
    unittest.main()
