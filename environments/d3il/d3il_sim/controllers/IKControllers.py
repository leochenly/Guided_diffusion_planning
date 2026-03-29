"""
This module contains the inverse kinematics controller.
"""

from abc import abstractmethod
import logging

import numpy as np

import environments.d3il.d3il_sim.controllers.GainsInterface as gains
import environments.d3il.d3il_sim.utils as utils
from environments.d3il.d3il_sim.controllers.Controller import TrackingController

import jax.numpy as jnp
from panda_kinematics import compute_franka_ik


class CartPosImpedenceController(TrackingController, gains.CartPosControllerConfig):
    """
    Controller for the cartesian coordinates of the robots end effector.
    """

    def __init__(self):

        TrackingController.__init__(self, dimSetPoint=3)
        gains.CartPosControllerConfig.__init__(self)

        self.J_reg = 1e-6  # Jacobian regularization constant
        self.W = np.diag([1, 1, 1, 1, 1, 1, 1])

        # Null-space theta configuration
        self.target_th_null = np.array(
            [
                3.57795216e-09,
                1.74532920e-01,
                3.30500960e-08,
                -8.72664630e-01,
                -1.14096181e-07,
                1.22173047e00,
                7.85398126e-01,
            ]
        )

        self.reset()

    def reset(self):
        self.desired_c_pos = np.array([0.624, 0, 0.55])
        self.desired_c_vel = np.zeros((3,))
        self.desired_c_acc = np.zeros((3,))

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        self.paramsLock.acquire()
        xd_d = self.desired_c_pos - robot.current_c_pos
        target_c_acc = self.pgain * xd_d

        J = robot.getJacobian()
        J = J[:3, :]
        Jw = J.dot(self.W)

        # J *  W * J' + reg * I
        JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(3)

        # Null space movement
        qd_null = self.pgain_null * (self.target_th_null - robot.current_j_pos)
        # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null

        qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
        qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel

        robot.jointTrackingController.setSetPoint(robot.current_j_pos, qd_d)
        self.paramsLock.release()

        return robot.jointTrackingController.getControl(robot)

    def setGains(self, pGain, dGain):
        """
        Setter for the gains of the PD Controller.

        :param pGain: p gain
        :param dGain: d gain
        :return: no return value
        """
        self.paramsLock.acquire()
        self.pgain = pGain
        self.dgain = dGain
        self.paramsLock.release()

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_pos = desired_pos
        if desired_vel is not None:
            self.desired_c_vel = desired_vel
        if desired_acc is not None:
            self.desired_c_acc = desired_acc
        self.paramsLock.release()

    def getCurrentPos(self, robot):
        """
        Getter for the robots current posi
        :param robot:
        :return:
        """
        return robot.current_c_pos

    def getDesiredPos(self, robot):
        return robot.des_c_pos


class CartesianPositionController(CartPosImpedenceController):
    def setAction(self, action):
        self.desired_c_pos = action

    @abstractmethod
    def reset(self):
        pass
 

class CartPosQuatImpedenceController(
    TrackingController, gains.CartPosQuatControllerConfig
):
    """
    Controller for the cartesian coordinates and the orientation (using quaternions) of the robots end effector.
    """

    def __init__(self):

        TrackingController.__init__(self, dimSetPoint=7)
        gains.CartPosQuatControllerConfig.__init__(self)
        self.W = np.diag(self.W)
        self.dgain_null = np.sqrt(self.pgain_null) * 2

        self.reset()
        # with this flag set on, the robot is always beamed to the correct location during the IK. Mainly used for debugging reasons (just test IK without dynamics)
        self.neglect_dynamics = False

    def reset(self):
        self.desired_c_pos = np.array([0.624, 0, 0.55])
        self.desired_quat = np.array([0.0, 0.984, 0, 0.177])
        self.desired_quat = self.desired_quat / np.linalg.norm(self.desired_quat)
        self.desired_c_vel = np.zeros((3,))
        self.desired_quat_vel = np.zeros((4,))

        self.old_des_joint_vel = np.zeros((7,))
        self.old_q = np.zeros((7,))
        self.old_q[:] = np.nan

    def getControl(self, robot):

        self.paramsLock.acquire()
        super(CartPosQuatImpedenceController, self).getControl(robot)

        if any(np.isnan(self.old_q)):
            self.old_q = robot.current_j_pos.copy()

        q = self.old_q.copy()

        q = (
            self.joint_filter_coefficient * q
            + (1 - self.joint_filter_coefficient) * robot.current_j_pos
        )

        qd_dsum = np.zeros(q.shape)

        oldErrorNorm = np.inf
        qd_d = np.zeros(q.shape)

        [current_c_pos, current_c_quat] = robot.getForwardKinematics(q)
        target_cpos_acc = self.desired_c_pos - current_c_pos

        curr_quat = current_c_quat

        if np.linalg.norm(curr_quat - self.desired_quat) > np.linalg.norm(
            curr_quat + self.desired_quat
        ):
            curr_quat = -curr_quat
        oldErrorNorm = np.sum(target_cpos_acc**2) + np.sum(
            (curr_quat - self.desired_quat) ** 2
        )

        des_quat = self.desired_quat
        for i in range(self.num_iter):

            [current_c_pos, current_c_quat] = robot.getForwardKinematics(q)
            target_cpos_acc = self.desired_c_pos - current_c_pos

            curr_quat = current_c_quat

            if np.linalg.norm(curr_quat - des_quat) > np.linalg.norm(
                curr_quat + des_quat
            ):
                des_quat = -des_quat
            errNorm = np.sum(target_cpos_acc**2) + np.sum((curr_quat - des_quat) ** 2)

            target_cquat = utils.get_quaternion_error(curr_quat, des_quat)

            target_cpos_acc = np.clip(target_cpos_acc, -0.01, 0.01)
            target_cquat = np.clip(target_cquat, -0.1, 0.1)
            # self.pgain_quat = np.zeros((3,))

            target_c_acc = np.hstack(
                (self.pgain_pos * target_cpos_acc, self.pgain_quat * target_cquat)
            )

            J = robot.getJacobian(q)

            # Singular Value decomposition, to clip the singular values which are too small/big

            Jw = J.dot(self.W)

            # J *  W * J' + reg * I
            condNumber = np.linalg.cond(Jw.dot(J.T))
            JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(J.shape[0])

            u, s, v = np.linalg.svd(JwJ_reg, full_matrices=False)
            s_orig = s
            s = np.clip(s, self.min_svd_values, self.max_svd_values)
            # reconstruct the Jacobian
            JwJ_reg = u @ np.diag(s) @ v
            condNumber2 = np.linalg.cond(JwJ_reg)
            largestSV = np.max(s_orig)

            qdev_rest = np.clip(self.rest_posture - q, -0.2, 0.2)

            # Null space movement
            qd_null = np.array(
                self.pgain_null * (qdev_rest)
            )  # + self.dgain_null * (-robot.current_j_vel)

            margin_to_limit = 0.01
            pgain_limit = 20

            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (
                robot.joint_pos_max - margin_to_limit - q
            )
            qd_null_limit_min = pgain_limit * (
                robot.joint_pos_min + margin_to_limit - q
            )
            qd_null_limit[
                q > robot.joint_pos_max - margin_to_limit
            ] += qd_null_limit_max[q > robot.joint_pos_max - margin_to_limit]
            qd_null_limit[
                q < robot.joint_pos_min + margin_to_limit
            ] += qd_null_limit_min[q < robot.joint_pos_min + margin_to_limit]

            # qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
            qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

            # clip desired joint velocities for stability

            if np.linalg.norm(qd_d) > 3:
                qd_d = qd_d * 3 / np.linalg.norm(qd_d)

            qd_dsum = qd_dsum + qd_d

            q = q + self.learningRate * qd_d
            q = np.clip(q, robot.joint_pos_min, robot.joint_pos_max)

        self.tracking_error = np.sum(np.abs(self.desired_c_pos - current_c_pos)) > 0.01

        qd_dsum = (q - self.old_q) / robot.dt
        des_acc = self.ddgain * (qd_dsum - self.old_des_joint_vel) / robot.dt

        if np.sum(np.abs(self.desired_c_pos - current_c_pos)) > 0.1:
            target_cquat = utils.get_quaternion_error(curr_quat, self.desired_quat)

            logging.getLogger(__name__).debug(
                "i: %d, Time: %f, Pos_error: %f, Quat_error: %f,  qd_d: %f"
                % (
                    i,
                    robot.time_stamp,
                    np.linalg.norm(self.desired_c_pos - current_c_pos),
                    np.linalg.norm(target_cquat),
                    np.linalg.norm(qd_d),
                ),
                qd_d,
                self.desired_c_pos,
                des_acc,
            )

        if np.linalg.norm(des_acc) > 10000:
            des_acc = des_acc * 10000 / np.linalg.norm(des_acc)

        robot.jointTrackingController.setSetPoint(q, qd_dsum, des_acc)
        # robot.jointTrackingController.setSetPoint(q, qd_dsum)#, des_acc)

        self.old_q = q.copy()
        self.old_des_joint_vel = qd_dsum

        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel
        robot.des_quat = self.desired_quat
        robot.des_quat_vel = self.desired_quat_vel
        robot.misc_data = np.array([errNorm, condNumber, condNumber2, largestSV])

        self.paramsLock.release()

        if self.neglect_dynamics:
            robot.beam_to_joint_pos(q, resetDesired=False)
            return np.zeros((7,))
        else:
            control = robot.jointTrackingController.getControl(robot)

            return control

    '''
    def setGains(self, pGain, dGain, pGain_null, dGain_null, dGainVelCtrl):
        """
        Setter for the gains of the PD Controller.

        :param pGain: p gain
        :param dGain: d gain
        :param pGain_null: p gain null
        :param dGain_null: d gain null
        :param dGainVelCtrl: gain for velocity control on top
        :return: no return value
        """
        # self.paramsLock.acquire()
        self.pgain = pGain
        self.dgain = dGain
        self.pgain_null = pGain_null
        self.dgain_null = dGain_null
        self.dgain_velcontroller = dGainVelCtrl
        # self.paramsLock.release()
    '''

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_pos = desired_pos[:3].copy()
        self.desired_quat = desired_pos[3:] / np.linalg.norm(desired_pos[3:])
        if desired_vel is not None:
            self.desired_c_vel = desired_vel[:3]
            self.desired_quat_vel = desired_vel[3:]

        self.paramsLock.release()

    def getCurrentPos(self, robot):
        """
        Getter for the robots current positions.

        :param robot: instance of the robot
        :return: current joint position (num_joints, 1)
        """
        return np.hstack((robot.current_c_pos, robot.current_c_quat))

    def getDesiredPos(self, robot):
        return np.concatenate((robot.des_c_pos, robot.des_quat))


class CartVelocityImpedenceController(CartPosQuatImpedenceController):
    """
    Controller for the cartesian coordinates and the orientation (using quaternions) of the robots end effector.
    """

    def __init__(self, fixed_orientation=None, max_cart_vel=0.5):

        super(CartVelocityImpedenceController, self).__init__()
        self.fixed_orientation = fixed_orientation
        self.max_cart_vel = max_cart_vel

        self.max_cart_pos = np.array([0.7, 0.4, 0.75])
        self.min_cart_pos = np.array([0.2, -0.4, 0.0])

    def getControl(self, robot):

        self.desired_c_pos = np.array(robot.current_c_pos) + robot.dt * np.array(
            self.desired_c_vel
        )
        self.desired_c_pos = np.clip(
            self.desired_c_pos, self.min_cart_pos, self.max_cart_pos
        )

        if self.fixed_orientation is None:
            self.desired_quat = self.fixed_orientation
        else:
            self.desired_quat = self.fixed_orientation

        return super(CartVelocityImpedenceController, self).getControl(robot)

    def setSetPoint(self, desired_pos, desired_vel, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_vel = desired_vel[:3]
        if self.fixed_orientation is None:
            self.desired_quat_vel = desired_vel[3:]

        self.paramsLock.release()


class CartPosQuatImpedenceJacTransposeController(
    CartPosQuatImpedenceController, gains.CartPosQuatJacTransposeControllerConfig
):
    def __init__(self):
        super(CartPosQuatImpedenceJacTransposeController, self).__init__()
        gains.CartPosQuatJacTransposeControllerConfig.__init__(self)

    def getControl(self, robot):
        self.paramsLock.acquire()
        qd_d = self.desired_c_pos - robot.current_c_pos
        target_cpos_acc = self.pgain_pos * qd_d
        curr_quat = (
            -robot.current_c_quat
            if (robot.current_c_quat @ self.desired_quat) < 0
            else robot.current_c_quat
        )
        target_cquat = self.pgain_quat * utils.get_quaternion_error(
            robot.current_c_quat, self.desired_quat
        )
        J = robot.getJacobian()

        qd_null = self.pgain_null * (
            self.rest_posture - robot.current_j_pos
        ) + self.dgain_null * (-robot.current_j_vel)
        target_acc = np.concatenate((target_cpos_acc, target_cquat))
        qdd = J.T @ (target_acc - self.dgain * (J @ robot.current_j_vel)) + qd_null
        # qdd = J.T@(self.pgain*pos_error + self.dgain*vel_error) + qd_null
        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel
        robot.des_quat = self.desired_quat
        robot.des_quat_vel = self.desired_quat_vel
        self.paramsLock.release()
        return qdd


class CartPosQuatCartesianRobotController(TrackingController):
    def __init__(self):
        TrackingController.__init__(self, dimSetPoint=7)
        self.reset()
    def reset(self):
        self.desired_pos = np.zeros((self.dimSetPoint,))

    def setAction(self, action):
        self.desired_pos = action  # should be named with quad

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        self.desired_pos = desired_pos

    def getControl(self, robot):
        robot.des_c_pos = self.desired_pos[:3]
        robot.des_c_vel = np.zeros((3,))
        if self.desired_pos.shape[0] > 3:
            robot.des_quat = self.desired_pos[3:]
        robot.des_quat_vel = np.zeros((4,))

        return self.desired_pos

    def getCurrentPos(self, robot):
        return np.hstack((robot.current_c_pos, robot.current_c_quat))

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    import mujoco
except Exception:
    mjx = None


import numpy as np

import environments.d3il.d3il_sim.controllers.GainsInterface as gains
import environments.d3il.d3il_sim.utils as utils
from environments.d3il.d3il_sim.controllers.Controller import TrackingController


class CartPosQuatImpedenceControllerMJX(
    TrackingController, gains.CartPosQuatControllerConfig
):
    """
    Differentiable MJX/JAX replica of CartPosQuatImpedenceController.getControl().

    - action is set via setSetPoint(): desired_c_pos (3), desired_quat (4)
    - getControl_mjx() writes torque commands into scene.mjx_data.ctrl via a fully JAX-traceable function
    - For full differentiability across time, use _compiled_stateful in your rollout and carry controller state in lax.scan.
    """

    def __init__(self):
        TrackingController.__init__(self, dimSetPoint=7)
        gains.CartPosQuatControllerConfig.__init__(self)

        self.W = np.diag(self.W)
        self.dgain_null = np.sqrt(self.pgain_null) * 2

        self.reset()
        self.neglect_dynamics = False

        # MJX/JAX compiled functions
        self._compiled = None
        self._compiled_stateful = None

        # OO-runtime MJX state 
        self._mjx_state = None

        # cached indices / ids
        self._mjx_cache_ready = False
        self.mjx_clip_actions = False

    def reset(self):
        # Keep CPU defaults
        self.desired_c_pos = np.array([0.624, 0, 0.55], dtype=np.float32)
        self.desired_quat = np.array([0.0, 0.984, 0, 0.177], dtype=np.float32)
        self.desired_quat = self.desired_quat / np.linalg.norm(self.desired_quat)
        self.desired_c_vel = np.zeros((3,), dtype=np.float32)
        self.desired_quat_vel = np.zeros((4,), dtype=np.float32)

        # For CPU parity (used as internal memory in the original controller)
        self.old_des_joint_vel = np.zeros((7,), dtype=np.float32)
        self.old_q = np.zeros((7,), dtype=np.float32)
        self.old_q[:] = np.nan

        # Reset MJX OO-state too
        self._mjx_state = None

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        desired_pos is expected to be 7D: [x,y,z,qw,qx,qy,qz]
        """
        self.paramsLock.acquire()
        desired_pos = np.asarray(desired_pos, dtype=np.float32)
        self.desired_c_pos = desired_pos[:3].copy()

        quat = desired_pos[3:].copy()
        n = np.linalg.norm(quat)
        if n < 1e-12:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat = quat / n
        self.desired_quat = quat

        if desired_vel is not None:
            desired_vel = np.asarray(desired_vel, dtype=np.float32)
            self.desired_c_vel = desired_vel[:3].copy()
            self.desired_quat_vel = desired_vel[3:].copy()
        self.paramsLock.release()

    # -------------------------
    # MJX build + compiled ctrl
    # -------------------------
    def _build(self, scene, robot):
        """
        Build jit-compiled JAX controller.
        """
        m = scene.mjx_model

        # ---- indices: use robot's own indices if available ----
        # Arm qpos / qvel indices
        if hasattr(robot, "joint_indices"):
            qpos_idx = np.asarray(robot.joint_indices, dtype=np.int32)
        elif hasattr(robot, "joint_pos_indices"):
            qpos_idx = np.asarray(robot.joint_pos_indices, dtype=np.int32)
        else:
            # fallback: assume first 7 qpos belong to arm
            qpos_idx = np.arange(7, dtype=np.int32)

        if hasattr(robot, "joint_vel_indices"):
            qvel_idx = np.asarray(robot.joint_vel_indices, dtype=np.int32)
        elif hasattr(robot, "joint_vel_indices"):
            qvel_idx = np.asarray(robot.joint_vel_indices, dtype=np.int32)
        else:
            # fallback: assume first 7 qvel belong to arm
            qvel_idx = np.arange(7, dtype=np.int32)

        # actuator indices for arm
        if hasattr(robot, "joint_act_indices"):
            act_idx = np.asarray(robot.joint_act_indices, dtype=np.int32)
        else:
            # fallback: first 7 actuators
            act_idx = np.arange(7, dtype=np.int32)

        # dt
        dt = float(getattr(robot, "dt", getattr(scene, "dt", 1e-3)))

        # ---- find tcp body id ----
        if robot.joint_indices is None:
            robot._init_jnt_indices()
        if not hasattr(robot, "joint_vel_indices") or robot.joint_vel_indices is None:
            robot.joint_vel_indices = robot.joint_indices

        tcp_name = robot.add_id2model_key("tcp")
        tcp_body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)
        if tcp_body_id == -1:
            raise RuntimeError(f"TCP body '{tcp_name}' not found in model.")

        tcp_site_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        if tcp_site_id == -1:
            raise RuntimeError(
                f"TCP site '{tcp_name}' not found in model. "
                f"CPU/compare path uses site Jacobian (mj_jacSite) and site xpos."
            )

        # ---- joint limits / torque limits ----
        q_min = jnp.asarray(getattr(robot, "joint_pos_min", [-np.inf] * 7), dtype=jnp.float32)
        q_max = jnp.asarray(getattr(robot, "joint_pos_max", [np.inf] * 7), dtype=jnp.float32)
        tau_limit = jnp.asarray(getattr(robot, "torque_limit", [np.inf] * 7), dtype=jnp.float32)

        # ---- controller params: copy from CPU config ----
        W = jnp.asarray(self.W, dtype=jnp.float32)  # (7,7)
        J_reg = jnp.asarray(self.J_reg, dtype=jnp.float32)
        min_svd = jnp.asarray(self.min_svd_values, dtype=jnp.float32)
        max_svd = jnp.asarray(self.max_svd_values, dtype=jnp.float32)
        learning_rate = jnp.asarray(self.learningRate, dtype=jnp.float32)
        num_iter = int(self.num_iter)
        joint_filter_coeff = jnp.asarray(self.joint_filter_coefficient, dtype=jnp.float32)
        ddgain = jnp.asarray(self.ddgain, dtype=jnp.float32)

        # Original CPU clips
        pos_clip = jnp.asarray(0.01, dtype=jnp.float32)
        rot_clip = jnp.asarray(0.1, dtype=jnp.float32)
        qd_norm_clip = jnp.asarray(3.0, dtype=jnp.float32)

        # task gains
        pgain_pos = jnp.asarray(self.pgain_pos, dtype=jnp.float32)     # (3,)
        pgain_quat = jnp.asarray(self.pgain_quat, dtype=jnp.float32)   # (3,)

        # nullspace (rest posture)
        rest_posture = getattr(self, "rest_posture", None)
        if rest_posture is None:
            rest_posture = np.zeros((7,), dtype=np.float32)
        rest_posture = jnp.asarray(rest_posture, dtype=jnp.float32)
        pgain_null = jnp.asarray(self.pgain_null, dtype=jnp.float32)

        # ---- execution layer: match CPU JointPDController if present ----
        # NOTE: In your project RobotBase.reset() reassigns jointTrackingController to JointPDController,
        # so this is typically what CPU uses at runtime.
        if hasattr(robot, "jointTrackingController") and hasattr(robot.jointTrackingController, "pgain"):
            pd_pgain = jnp.asarray(robot.jointTrackingController.pgain, dtype=jnp.float32)
            pd_dgain = jnp.asarray(robot.jointTrackingController.dgain, dtype=jnp.float32)
        else:
            # fallback gains
            pd_pgain = jnp.ones((7,), dtype=jnp.float32) * 100.0
            pd_dgain = jnp.ones((7,), dtype=jnp.float32) * 10.0

        # -----------------
        # Helpers (all JAX)
        # -----------------
        def align_des_quat(curr_quat, des_quat):
            # CPU logic: flip desired quat if that makes it closer to current
            n1 = jnp.linalg.norm(curr_quat - des_quat)
            n2 = jnp.linalg.norm(curr_quat + des_quat)
            return jax.lax.cond(n1 > n2, lambda q: -q, lambda q: q, des_quat)

        def quat_err_siciliano(curr_quat, des_quat):
            # 3D quaternion error used in CPU controller (Siciliano et al.)
            cq0, cq1, cq2, cq3 = curr_quat
            dq0, dq1, dq2, dq3 = des_quat
            e0 = cq0 * dq1 - dq0 * cq1 - cq3 * dq2 + cq2 * dq3
            e1 = cq0 * dq2 - dq0 * cq2 + cq3 * dq1 - cq1 * dq3
            e2 = cq0 * dq3 - dq0 * cq3 - cq2 * dq1 + cq1 * dq2
            return jnp.array([e0, e1, e2], dtype=curr_quat.dtype)

        def _vee(skew_mat):
            return jnp.array([skew_mat[2, 1], skew_mat[0, 2], skew_mat[1, 0]], dtype=skew_mat.dtype)

        def _site_pos_R_and_body_quat(d_in, qpos_full):
            """
            Forward kinematics at arbitrary qpos_full:
            - position: SITE xpos (matches mj_jacSite point)
            - orientation: SITE xmat for Jacobian rotation (angular vel)
            - quaternion for error: BODY xquat (matches compare + likely CPU state)
            """
            d_tmp = d_in.replace(qpos=qpos_full)
            d_f = mjx.forward(m, d_tmp)

            pos_site = d_f.site_xpos[tcp_site_id]  # (3,)
            R_site = d_f.site_xmat[tcp_site_id].reshape(3, 3)  # (3,3) robust for (9,) or (3,3)
            quat_body = d_f.xquat[tcp_body_id]  # (4,) [w,x,y,z]
            return pos_site, R_site, quat_body, d_f

        def tcp_pose_from_qpos(d_in, qpos_full):
            """
            What the controller uses:
            - pos from SITE
            - quat from BODY
            """
            pos, _, quat, d_f = _site_pos_R_and_body_quat(d_in, qpos_full)
            return pos, quat, d_f

        def make_task_and_jac(d_in, qpos_full, target_pos, target_quat):
            # FK
            pos, R, quat, _ = _site_pos_R_and_body_quat(d_in, qpos_full)

            # desired quat sign alignment (CPU behavior)
            desq = align_des_quat(quat, target_quat)

            # task error (same as your existing version)
            pos_err = jnp.clip(target_pos - pos, -pos_clip, pos_clip)
            rot_err = jnp.clip(quat_err_siciliano(quat, desq), -rot_clip, rot_clip)
            target_task = jnp.concatenate([pgain_pos * pos_err, pgain_quat * rot_err], axis=0)  # (6,)

            # -----------------------
            # Jacobian: J = [Jp; Jr]
            # -----------------------
            # Jp: d(site_pos)/dq
            Jp_full = jax.jacrev(lambda qf: _site_pos_R_and_body_quat(d_in, qf)[0])(qpos_full)  # (3,nq)

            # Jr: angular-velocity Jacobian for the SITE frame.
            # Construct from R(q): for each qi, S_i = (dR/dqi) * R^T, omega_i = vee( 0.5*(S_i - S_i^T) )
            def _R_flat(qf):
                return _site_pos_R_and_body_quat(d_in, qf)[1].reshape(-1)

            dR_dq = jax.jacrev(_R_flat)(qpos_full).reshape(3, 3, -1)  # (3,3,nq)
            RT = R.T

            def _omega_col(i):
                dR_i = dR_dq[:, :, i]
                S = dR_i @ RT
                # project to skew-symmetric to reduce numerical asymmetry
                S = 0.5 * (S - S.T)
                return _vee(S)  # (3,)

            Jr_full = jnp.stack([_omega_col(i) for i in range(dR_dq.shape[2])], axis=1)  # (3,nq)

            # slice to arm joints
            Jp = Jp_full[:, qpos_idx]  # (3,7)
            Jr = Jr_full[:, qpos_idx]  # (3,7)
            J = jnp.vstack([Jp, Jr])   # (6,7)

            return target_task, J
        # ----------------------
        # IK iteration (scan body) — FIXED: no free variables
        # carry = (q_arm, d_in, target_pos, target_quat)
        # ----------------------
        def ik_iter(carry, _):
            q_arm, d_local, target_pos_local, target_quat_local = carry

            # full qpos with updated arm
            qpos_full = d_local.qpos.at[qpos_idx].set(q_arm)

            target_task, J = make_task_and_jac(d_local, qpos_full, target_pos_local, target_quat_local)

            # nullspace: rest posture only (CPU joint-limit term is commented out in your version)
            qdev_rest = jnp.clip(rest_posture - q_arm, -0.2, 0.2)
            qd_null = pgain_null * qdev_rest  # (7,)

            # CPU DLS with weighting
            Jw = J @ W
            A = Jw @ J.T + J_reg * jnp.eye(6, dtype=jnp.float32)

            # CPU SVD clip on A
            u, s, vt = jnp.linalg.svd(A, full_matrices=False)
            s = jnp.clip(s, min_svd, max_svd)
            A_clipped = (u * s) @ vt

            rhs = target_task - (J @ qd_null)
            x = jnp.linalg.solve(A_clipped, rhs)
            qd = (W @ J.T) @ x + qd_null  # (7,)

            # norm clip like CPU
            n = jnp.linalg.norm(qd)
            qd = jnp.where(n > qd_norm_clip, qd * (qd_norm_clip / (n + 1e-12)), qd)

            # update q
            q_new = q_arm + learning_rate * qd
            q_new = jnp.clip(q_new, q_min, q_max)

            new_carry = (q_new, d_local, target_pos_local, target_quat_local)
            return new_carry, qd

        # ----------------------
        # Stateful ctrl_fn (for full differentiability across time)
        # state = (old_q, old_des_joint_vel), each (7,)
        # ----------------------
        def ctrl_fn_stateful(m, d_in, target_pos, target_quat, state):
            old_q, old_des_joint_vel = state

            # current arm states
            q_arm = d_in.qpos[qpos_idx]
            qvel_arm = d_in.qvel[qvel_idx]

            # if old_q is nan (first call), initialize with current q
            old_q_init = jnp.where(jnp.any(jnp.isnan(old_q)), q_arm, old_q)

            # CPU-like joint filtering
            q_init = joint_filter_coeff * old_q_init + (1.0 - joint_filter_coeff) * q_arm

            # IK scan
            # q_final, _qd_seq = jax.lax.scan(ik_iter, q_init, xs=None, length=num_iter)
            carry0 = (q_init, d_in, target_pos, target_quat)
            (carryN, _qd_seq) = jax.lax.scan(ik_iter, carry0, xs=None, length=num_iter)
            q_final = carryN[0]

            # CPU-like qd & qdd construction
            qd_des = (q_final - old_q_init) / dt
            qdd_des = ddgain * (qd_des - old_des_joint_vel) / dt  # kept for parity (not strictly needed here)

            # execution layer: JointPD torque-like command
            tau = pd_pgain * (q_final - q_arm) + pd_dgain * (qd_des - qvel_arm)

            # gravity/coriolis bias (aligns with CPU gravity_comp branch semantics)
            d_f = mjx.forward(m, d_in)
            tau = tau + d_f.qfrc_bias[qvel_idx]

            # clip like preprocessCommand (torque_limit)
            # tau = jnp.clip(tau, -tau_limit, tau_limit)
            if getattr(self, "mjx_clip_actions", False):
                tau = jnp.clip(tau, -tau_limit, tau_limit)

            # write ctrl
            ctrl = d_in.ctrl.at[act_idx].set(tau)
            d_out = d_in.replace(ctrl=ctrl)

            new_state = (q_final, qd_des)
            return d_out, new_state

        # Stateless wrapper (useful for quick tests; not identical to CPU due to missing memory)
        def ctrl_fn(m, d_in, target_pos, target_quat):
            old_q0 = jnp.full((7,), jnp.nan, dtype=jnp.float32)
            old_v0 = jnp.zeros((7,), dtype=jnp.float32)
            d_out, _ = ctrl_fn_stateful(m, d_in, target_pos, target_quat, (old_q0, old_v0))
            return d_out

        # compile
        self._compiled_stateful = jax.jit(ctrl_fn_stateful)
        self._compiled = jax.jit(ctrl_fn)

        # cache everything needed by getControl_mjx
        self._mjx_cache_ready = True

    def getControl_mjx(self, scene, robot):
        """
        OO runtime helper:
          - reads desired setpoint from python fields
          - updates scene.mjx_data.ctrl
        This is differentiable inside MJX rollout only if you use _compiled_stateful in your own lax.scan carry.
        """
        if self._compiled_stateful is None or not self._mjx_cache_ready:
            self._build(scene, robot)

        import jax.numpy as jnp

        # read setpoint
        self.paramsLock.acquire()
        desired_c_pos = np.asarray(self.desired_c_pos, dtype=np.float32).copy()
        desired_quat = np.asarray(self.desired_quat, dtype=np.float32).copy()
        self.paramsLock.release()

        # normalize quat
        n = float(np.linalg.norm(desired_quat))
        if n < 1e-12:
            desired_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            desired_quat = desired_quat / n

        target_pos = jnp.asarray(desired_c_pos, dtype=jnp.float32)
        target_quat = jnp.asarray(desired_quat, dtype=jnp.float32)

        if self._mjx_state is None:
            old_q = jnp.full((7,), jnp.nan, dtype=jnp.float32)
            old_v = jnp.zeros((7,), dtype=jnp.float32)
            self._mjx_state = (old_q, old_v)

        scene.mjx_data, self._mjx_state = self._compiled_stateful(
            scene.mjx_model, scene.mjx_data, target_pos, target_quat, self._mjx_state
        )
        return scene.mjx_data



