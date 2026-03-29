import os
import xml.etree.ElementTree as Et
from typing import Tuple

import mujoco
import numpy as np 

from environments.d3il.d3il_sim.controllers.Controller import ModelBasedFeedforwardController
from environments.d3il.d3il_sim.core import RobotBase
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import (
    IncludeType,
    get_body_xvelp,
    get_body_xvelr,
)
from environments.d3il.d3il_sim.sims.mj_beta.MjCamera import MjInhandCamera
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import (
    MjIncludeTemplate,
    MjXmlLoadable,
)
from environments.d3il.d3il_sim.utils import sim_path

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
except Exception:
    jax, jnp, mjx = None, None, None

class MjRobot(RobotBase, MjIncludeTemplate):
    GLOBAL_MJ_ROBOT_COUNTER = 0

    def __init__(
        self,
        scene,
        dt=1e-3,
        num_DoF=7,
        base_position=None,
        base_orientation=None,
        gravity_comp=True,
        clip_actions=False,
        root=sim_path.D3IL_DIR,
        xml_path=None,
    ):
        RobotBase.__init__(self, scene, dt, num_DoF, base_position, base_orientation)

        if xml_path is None:
            xml_path = sim_path.d3il_path("./models/mj/robot/panda.xml")
        MjXmlLoadable.__init__(self, xml_path)

        self.clip_actions = clip_actions
        self.gravity_comp = gravity_comp

        self.joint_names = None
        self.joint_indices = None
        self.joint_act_indices = None

        self.gripper_names = None
        self.gripper_indices = None
        self.gripper_act_indices = None

        self.jointTrackingController = ModelBasedFeedforwardController()

        # Global "unique" ID for multibot support
        self._mj_robot_id = MjRobot.GLOBAL_MJ_ROBOT_COUNTER
        MjRobot.GLOBAL_MJ_ROBOT_COUNTER += 1
        self.root = root

        self.inhand_cam = MjInhandCamera(self.add_id2model_key("rgbd"))

        # Position and velocity for freezing operations
        self.qpos_freeze = None
        self.qvel_freeze = None
        self.ctrl_freeze = None

    def _getJacobian_internal(self, q=None):
        """
        Differentiable Jacobian using autodiff on FK.
        Output shape: (6, 7) [pos; rot] like old code. :contentReference[oaicite:9]{index=9}
        """
        if q is None:
            q = self.current_j_pos

        q = jnp.asarray(q)

        # FK function: q -> (pos, quat)
        def fk_pos(q_in):
            pos, quat = self.getForwardKinematics(q_in)
            return pos

        # Position Jacobian (3,7)
        J_pos = jax.jacrev(fk_pos)(q)

        # For rotational part, we linearize quaternion error:
        # use small-angle proxy e = quat_vec_part after aligning sign
        def rot_err_vec(q_in):
            _, quat = self.getForwardKinematics(q_in)
            # align sign to avoid discontinuity
            quat = jnp.where(quat[0] < 0.0, -quat, quat)
            # return vector part (x,y,z) as local orientation coordinate
            return quat[1:]

        J_rot3 = jax.jacrev(rot_err_vec)(q)  # (3,7)

        J = jnp.concatenate([J_pos, J_rot3], axis=0)
        return np.array(J) if isinstance(q, (np.ndarray,)) else J

    def _getForwardKinematics_internal(self, q=None):
        """
        FK from mjx_data (differentiable).
        Returns (pos, quat) in WORLD frame (same semantics as old get_body_xpos/xquat).
        """
        # If not using mjx, fall back to old numpy mujoco code path
        if (not hasattr(self.scene, "mjx_data")) or (self.scene.mjx_data is None) or (mjx is None):
            # ---- keep your old implementation here if you want fallback ----
            if q is not None:
                cur_sim_state = self.scene.sim.get_state()
                qpos_idx = self.joint_indices
                self.scene.data.qpos[qpos_idx] = q
                mujoco.mj_kinematics(self.scene.model, self.scene.data)
            tcp_id = self.add_id2model_key("tcp")
            cart_pos = self.scene.data.get_body_xpos(tcp_id)
            cart_or = self.scene.data.get_body_xquat(tcp_id)
            if q is not None:
                self.scene.sim.set_state(cur_sim_state)
            return cart_pos, cart_or

        # --- mjx path ---
        if self.joint_indices is None:
            self._init_jnt_indices()

        # Create a temporary mjx_data with q substituted if provided
        d = self.scene.mjx_data
        if q is not None:
            q = jnp.asarray(q)
            qpos = d.qpos.at[jnp.asarray(self.joint_indices)].set(q)
            d = d.replace(qpos=qpos)

        # forward kinematics
        d_f = mjx.forward(self.scene.mjx_model, d)

        # tcp is a BODY in your current code (`get_body_xpos/xquat`) :contentReference[oaicite:8]{index=8}
        tcp_name = self.add_id2model_key("tcp")
        tcp_bid = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)

        pos = d_f.xpos[tcp_bid]
        quat = d_f.xquat[tcp_bid]  # MuJoCo uses [w,x,y,z]
        return pos, quat
    
    def prepare_step(self):
        """
        Prepare one control write before physics step.

        CPU mode (legacy):
        activeController.getControl(self) -> preprocessCommand -> write self.scene.data.ctrl -> receiveState()

        MJX mode:
        A) Preferred: activeController.getControl_mjx(scene, robot) writes self.scene.mjx_data.ctrl (pure JAX controller path)
        B) Fallback:  legacy numpy controller -> preprocessCommand -> write BOTH mjx_data.ctrl and data.ctrl (bridge path)

        Notes
        -----
        - In MJX mode we do NOT call receiveState() here. State should be updated after physics stepping.
        - This function is for the d3il OO stepping chain. Your pure differentiable rollout should bypass this and call
        controller._compiled(...) + mjx.step(...) inside a JAX scan.
        """
        if self.joint_indices is None:
            self._init_jnt_indices()

        use_mjx = bool(getattr(self.scene, "use_mjx", False)) and (mjx is not None) \
                and (getattr(self.scene, "mjx_model", None) is not None) \
                and (getattr(self.scene, "mjx_data", None) is not None)

        # ---------- Legacy CPU path ----------
        if not use_mjx:
            self.command = self.activeController.getControl(self)
            self.preprocessCommand(self.command)

            self.scene.data.ctrl[self.joint_act_indices] = self.uff.copy()[: self.num_DoF]
            if len(self.gripper_act_indices) > 0:
                self.scene.data.ctrl[self.gripper_act_indices] = self.finger_commands

            # legacy behavior
            self.receiveState()
            return

        # ---------- MJX path ----------
        # cache device indices once
        if not hasattr(self, "_mjx_joint_act_idx") or self._mjx_joint_act_idx is None:
            self._mjx_joint_act_idx = jnp.asarray(self.joint_act_indices, dtype=jnp.int32)
            self._mjx_gripper_act_idx = jnp.asarray(self.gripper_act_indices, dtype=jnp.int32)

        # Preferred: pure-JAX controller writes scene.mjx_data.ctrl directly
        if hasattr(self.activeController, "getControl_mjx"):
            # Convention: getControl_mjx returns updated mjx_data (or mutates scene.mjx_data and returns it)
            out = self.activeController.getControl_mjx(self.scene, self)

            if out is not None:
                self.scene.mjx_data = out

            # Optional bridge sync of ctrl for debugging / logging / render-consistency
            if getattr(self.scene, "data", None) is not None and getattr(self.scene.model, "nu", 0) > 0:
                try:
                    self.scene.data.ctrl[:] = np.asarray(self.scene.mjx_data.ctrl)
                except Exception:
                    pass
            return

        # Fallback: use old numpy controller (NOT differentiable through controller)
        self.command = self.activeController.getControl(self)
        self.preprocessCommand(self.command)

        ctrl = self.scene.mjx_data.ctrl
        ctrl = ctrl.at[self._mjx_joint_act_idx].set(jnp.asarray(self.uff[: self.num_DoF]))

        if self._mjx_gripper_act_idx.size > 0:
            ctrl = ctrl.at[self._mjx_gripper_act_idx].set(jnp.asarray(self.finger_commands))

        self.scene.mjx_data = self.scene.mjx_data.replace(ctrl=ctrl)

        # Bridge sync to CPU ctrl for legacy code that inspects scene.data.ctrl
        if getattr(self.scene, "data", None) is not None and getattr(self.scene.model, "nu", 0) > 0:
            self.scene.data.ctrl[self.joint_act_indices] = np.asarray(self.uff[: self.num_DoF])
            if len(self.gripper_act_indices) > 0:
                self.scene.data.ctrl[self.gripper_act_indices] = np.asarray(self.finger_commands)

        # IMPORTANT: no receiveState() here in MJX path (state updates after physics step)

    def receiveState(self):
        if self.joint_indices is None:
            self._init_jnt_indices()

        tcp_name = self.add_id2model_key("tcp")

        ### JOINT STATE
        self.current_j_pos = np.array(
            [self.scene.data.joint(name).qpos.copy() for name in self.joint_names]
        ).squeeze()
        self.current_j_vel = np.array(
            [self.scene.data.joint(name).qvel.copy() for name in self.joint_names]
        ).squeeze()

        test = self.scene.data.body(tcp_name)
        ### ENDEFFECTOR GLOBAL
        self.current_c_pos_global = self.scene.data.body(tcp_name).xpos.copy()
        self.current_c_vel_global = get_body_xvelp(
            self.scene.model,
            self.scene.data,
            tcp_name,
        )
        self.current_c_quat_global = self.scene.data.body(tcp_name).xquat.copy()
        self.current_c_quat_vel_global = np.zeros(4)

        self.current_c_quat_vel_global[1:] = get_body_xvelr(
            self.scene.model, self.scene.data, tcp_name
        )
        self.current_c_quat_vel_global *= 0.5 * self.current_c_quat_global

        ### ENDEFFECTOR LOCAL
        self.current_c_pos, self.current_c_quat = self._localize_cart_coords(
            self.current_c_pos_global, self.current_c_quat_global
        )
        self.current_c_vel, _ = self._localize_cart_coords(
            # add base_position, as it is subtracted in _localize_cart_coords
            self.current_c_vel_global
            + self.base_position
        )
        # This must be checked!
        _, self.current_c_quat_vel = self._localize_cart_coords(
            self.base_position, self.current_c_quat_vel_global
        )

        ### FINGER STATE
        self.current_fing_pos = np.array(
            [self.scene.data.joint(name).qpos.copy() for name in self.gripper_names]
        ).squeeze()
        self.current_fing_vel = np.array(
            [self.scene.data.joint(name).qvel.copy() for name in self.gripper_names]
        ).squeeze()
        self.gripper_width = self.current_fing_pos[-2] + self.current_fing_pos[-1]
    
    def get_command_from_inverse_dynamics(self, target_j_acc, mj_calc_inv=False):
        """
        Differentiable inverse dynamics / bias via mjx.
        Fallback to numpy mujoco if mjx not available.
        """
        if (not hasattr(self.scene, "mjx_data")) or (self.scene.mjx_data is None) or (mjx is None):
            # ---- original implementation ---- :contentReference[oaicite:12]{index=12}
            if mj_calc_inv:
                self.scene.data.qacc[self.joint_vel_indices + self.gripper_vel_indices] = target_j_acc
                mujoco.mj_inverse(self.scene.model, self.scene.sim.data)
                return self.scene.data.qfrc_inverse[self.joint_vel_indices + self.gripper_vel_indices]
            else:
                return self.scene.data.qfrc_bias[self.joint_vel_indices + self.gripper_vel_indices]

        if self.joint_indices is None:
            self._init_jnt_indices()

        d = self.scene.mjx_data

        # forward to update bias
        d_f = mjx.forward(self.scene.mjx_model, d)

        if not mj_calc_inv:
            bias = d_f.qfrc_bias
            idx = jnp.asarray(self.joint_vel_indices + self.gripper_vel_indices)
            return bias[idx]

        # inverse dynamics: depends on mjx version; try mjx.inverse if exists
        if hasattr(mjx, "inverse"):
            # set qacc on dof indices
            qacc = d_f.qacc.at[jnp.asarray(self.joint_vel_indices + self.gripper_vel_indices)].set(jnp.asarray(target_j_acc))
            d_i = d_f.replace(qacc=qacc)
            d_i = mjx.inverse(self.scene.mjx_model, d_i)
            frc = d_i.qfrc_inverse
            idx = jnp.asarray(self.joint_vel_indices + self.gripper_vel_indices)
            return frc[idx]

        # fallback: use bias only
        idx = jnp.asarray(self.joint_vel_indices + self.gripper_vel_indices)
        return d_f.qfrc_bias[idx]

    def get_init_qpos(self):
        return np.array(
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

    def _init_jnt_indices(self):
        """Initialize relevant joint and actuator names and indices"""
        n_joints = len(self.joint_names)
        assert (
            n_joints == self.num_DoF
        ), "Error, found {} joints, but expected {}".format(n_joints, self.num_DoF)
        assert (
            len(self.gripper_names) == 2
        ), "Error, found more gripper joints than expected."

        self.joint_indices = []
        self.joint_vel_indices = []
        self.gripper_indices = []
        self.gripper_vel_indices = []
        for jnt_name in self.joint_names:
            jnt_id = mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_JOINT, name=jnt_name
            )
            self.joint_indices.append(self.scene.model.jnt_qposadr[jnt_id])
            self.joint_vel_indices.append(self.scene.model.jnt_dofadr[jnt_id])
        for grp_name in self.gripper_names:
            grp_id = mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_JOINT, name=grp_name
            )
            self.gripper_indices.append(self.scene.model.jnt_qposadr[grp_id])
            self.gripper_vel_indices.append(self.scene.model.jnt_dofadr[grp_id])
        self.joint_act_indices = [
            mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=name + "_act"
            )
            for name in self.joint_names
        ]
        self.gripper_act_indices = [
            mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=name + "_act"
            )
            for name in self.gripper_names
        ]

    def set_q(self, joint_pos):
        """
        Sets the value of the robot joints.
        Args:
            joint_pos: Value for the robot joints.

        Returns:
            No return value
        """
        if self.joint_indices is None:
            self._init_jnt_indices()

        qpos = self.scene.data.qpos.copy()
        qpos[self.joint_indices] = joint_pos

        qvel = self.scene.data.qvel.copy()
        # Use len() as qvel somehow can be shorter than qpos??
        qvel[self.joint_vel_indices] = np.zeros(len(self.joint_vel_indices))
        self.scene.set_state(qpos, qvel)

    def add_id2model_key(self, model_key_id: str) -> str:
        """modifies a model key identifier to include the robot id

        Args:
            model_key_id (str): an identifier of the xml model, e.g. joint1

        Returns:
            str: model_key with appended id, e.g. joint1_1
        """
        attrib_split = model_key_id.split("_")
        attrib_split.insert(1, "rb{}".format(self._mj_robot_id))
        attrib_id = "_".join(attrib_split)
        return attrib_id

    def modify_template(self, et: Et.ElementTree) -> str:
        self.joint_names = None
        self.joint_indices = None
        self.joint_act_indices = None

        self.gripper_names = None
        self.gripper_indices = None
        self.gripper_act_indices = None

        for node in et.iter():
            for attrib in [
                "body1",
                "body2",
                "name",
                "class",
                "childclass",
                "mesh",
                "site",
                "joint",
            ]:
                if attrib in node.attrib:
                    model_key_id = node.get(attrib)
                    attrib_id = self.add_id2model_key(model_key_id)
                    node.set(attrib, attrib_id)

        wb = et.find("worldbody")
        body_root = wb.find("body")
        body_root.set("pos", " ".join(map(str, self.base_position)))
        body_root.set("quat", " ".join(map(str, self.base_orientation)))

        self.joint_names = []
        self.gripper_names = []
        for jnt in wb.iter("joint"):
            if jnt.get("name") is not None and jnt.get("name").startswith(
                self.add_id2model_key("panda_joint")
            ):
                self.joint_names.append(jnt.get("name"))
            if jnt.get("name") is not None and jnt.get("name").startswith(
                self.add_id2model_key("panda_finger_joint")
            ):
                self.gripper_names.append(jnt.get("name"))

        import uuid

        new_path = sim_path.d3il_path(
            f"./models/mj/robot/panda_tmp_rb{self._mj_robot_id}_{uuid.uuid1()}.xml"
        )
        et.write(new_path)
        return new_path