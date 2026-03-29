import itertools
import logging
import os
from typing import List, Tuple

USE_MJX_DEFAULT = True

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
except Exception:
    jax, jnp, mjx = None, None, None


import mujoco
import numpy as np
from mujoco import MjData, MjModel, mj_name2id, mjtObj

from environments.d3il.d3il_sim.core.Scene import Scene
from environments.d3il.d3il_sim.core.sim_object.sim_object import IntelligentSimObject, SimObject
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
    reset_singleton as reset_render_singleton,
)
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_renderer import Viewer
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_scene_object import MjSurrounding
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_scene_parser import MjSceneParser
from environments.d3il.d3il_sim.sims.mj_beta.MjCamera import MjCageCam, MjCamera
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjFreezable, MjXmlLoadable
from environments.d3il.d3il_sim.utils import sim_path 


class MjScene(Scene):
    def __init__(
        self,
        object_list=None,
        dt=0.001,
        render=Scene.RenderMode.HUMAN,
        surrounding=None,
        random_env=False,
        main_xml_path=None,
    ):

        super(MjScene, self).__init__(object_list=object_list, dt=dt, render=render)

        self.data: MjData = None
        self.model: MjModel = None
        self.viewer: Viewer = None
        self.rt_obj = []
        
        self.use_mjx = USE_MJX_DEFAULT
        # self.use_mjx = False
        self.use_mjx_runtime_step = True
        self.mjx_model = None
        self.mjx_data = None
        self._mjx_step_jit = None

        self.init_qpos, self.init_qvel = None, None
        self.random_env = random_env

        if surrounding is None:
            surrounding = sim_path.d3il_path(
                "./models/mujoco/surroundings/lab_surrounding2.xml"
            )

        self.surrounding = MjSurrounding(
            surrounding_name=os.path.split(os.path.splitext(surrounding)[0])[-1],
            root=os.path.dirname(surrounding),
        )

        self.mj_scene_parser = MjSceneParser(main_xml_path=main_xml_path)

        self.cage_cam = MjCageCam()
        self.add_object(self.cage_cam)

    @property
    def sim_name(self) -> str:
        return "mj"

    def _setup_scene(self):
        for rb in self.robots:
            self.add_object(rb.inhand_cam)

        self.model, self.data = self.mj_scene_parser.create_scene(
            self.robots, self.surrounding, self.obj_repo.get_obj_list(), self.dt
        )

        self.viewer = None
        if self.render_mode == self.RenderMode.HUMAN:
            self.viewer = Viewer(self.model, self.data)  # Renderer
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = np.zeros(self.data.qvel.shape)

        self._setup_objects(self.obj_repo.get_obj_list())
        
        if self.use_mjx and (mjx is not None):
            import mujoco as mj
            self.model.geom_type[self.model.geom_type == mj.mjtGeom.mjGEOM_CYLINDER] = mj.mjtGeom.mjGEOM_CAPSULE
            self.mjx_model = mjx.put_model(self.model)
            self.mjx_data = mjx.make_data(self.mjx_model)
            self.mjx_data = self.mjx_data.replace(
                qpos=jnp.asarray(self.data.qpos),
                qvel=jnp.asarray(self.data.qvel),
                act=jnp.asarray(self.data.act) if self.model.na > 0 else self.mjx_data.act,
                ctrl=jnp.asarray(self.data.ctrl) if self.model.nu > 0 else self.mjx_data.ctrl,
            )

    def set_views(self, views):
        if self.render_mode == self.RenderMode.HUMAN:
            cam_ids = []
            for v in views:
                cam_ids.append(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, v.name)
                )
            self.viewer.set_cameras(cam_ids)

    def load_robot_to_scene(self, robot_init_qpos: np.ndarray = None):
        """
        Sets the initial joint position of the panda robot.

        Args:
            robot_init_qpos: numpy array (num dof,); initial joint positions

        Returns:
            No return value 
        """
        if robot_init_qpos is None:
            robot_init_qpos = np.stack([robot.get_init_qpos() for robot in self.robots])
        else:
            robot_init_qpos = np.asarray(robot_init_qpos)
        # Check input dimensionality, in case of legacy code with only one robot
        if robot_init_qpos.ndim == 1:
            robot_init_qpos = np.expand_dims(robot_init_qpos, 0)

        for i in range(len(self.robots)):
            self.robots[i].beam_to_joint_pos(robot_init_qpos[i], run=False)
        self.init_qpos = self.data.qpos.copy()
    
    def _sim_step(self):
        """
        Legacy OOP scene runtime step.

        Default behavior:
            - CPU MuJoCo step (stable for env.reset/env.step)

        Optional behavior (only if explicitly enabled):
            - MJX runtime step (for experiments)
              NOTE: this is NOT recommended for the legacy OOP env path during debugging.
        """
        can_use_mjx_runtime = (
            bool(getattr(self, "use_mjx", False))
            and bool(getattr(self, "use_mjx_runtime_step", False))
            and (mjx is not None)
            and (jax is not None)
            and (getattr(self, "mjx_model", None) is not None)
            and (getattr(self, "mjx_data", None) is not None)
        )

        # ===== Default / safe path: CPU MuJoCo =====
        if not can_use_mjx_runtime:
            mujoco.mj_step(self.model, self.data)
            print("using CPU Mujoco step")
            return

        # ===== Optional MJX runtime path =====
        # Lazy-compile the MJX step function
        if self._mjx_step_jit is None:
            self._mjx_step_jit = jax.jit(lambda d: mjx.step(self.mjx_model, d))

        # Bridge CPU ctrl -> MJX ctrl (legacy controllers often write to self.data.ctrl)
        if getattr(self.model, "nu", 0) > 0:
            try:
                self.mjx_data = self.mjx_data.replace(ctrl=jnp.asarray(self.data.ctrl))
            except Exception:
                # Fallback silently; if replace fails, MJX step may still use existing ctrl
                pass

        # Run one MJX step
        self.mjx_data = self._mjx_step_jit(self.mjx_data)
        # print("using MJX step")

        # Sync MJX state back to CPU mujoco.MjData for legacy receiveState/render/logging
        try:
            self.data.qpos[:] = np.asarray(self.mjx_data.qpos)
            self.data.qvel[:] = np.asarray(self.mjx_data.qvel)

            if getattr(self.model, "na", 0) > 0 and getattr(self.mjx_data, "act", None) is not None:
                self.data.act[:] = np.asarray(self.mjx_data.act)

            if getattr(self.model, "nu", 0) > 0 and getattr(self.mjx_data, "ctrl", None) is not None:
                self.data.ctrl[:] = np.asarray(self.mjx_data.ctrl)

            # refresh kinematics/derived quantities on CPU side
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            # As a last resort, avoid crashing the whole loop due to sync issues
            pass

    def sim_steps(self, n_steps):
        mujoco.mj_step(self.model, self.data, n_steps)

    def render(self):
        if self.render_mode == self.RenderMode.HUMAN:
            self.viewer.render()

    def reset(self, obj_pos=None):
        """Resets the scene (including the robot) to the initial conditions."""
        if obj_pos is None:
            obj_pos = []

        for rb in self.robots:
            rb.reset()

        mujoco.mj_resetData(self.model, self.data)

        # Set initial position and velocity
        qpos = self.data.qpos.copy()
        qpos[:] = self.init_qpos

        qvel = np.zeros(self.data.qvel.shape)
        self.set_state(qpos, qvel)

        for (obj, new_pos) in obj_pos:
            self.set_obj_pos(new_pos, obj)

        mujoco.mj_forward(self.model, self.data)
        
        # if self.use_mjx:
        #     self.mjx_data = mjx.put_data(self.model, self.data)
        
        if self.use_mjx and (mjx is not None) and (self.mjx_model is not None):
            if self.mjx_data is None:
                self.mjx_data = mjx.make_data(self.mjx_model)
            self.mjx_data = self.mjx_data.replace(
                qpos=jnp.asarray(self.data.qpos),
                qvel=jnp.asarray(self.data.qvel),
                act=jnp.asarray(self.data.act) if self.model.na > 0 else self.mjx_data.act,
                ctrl=jnp.asarray(self.data.ctrl) if self.model.nu > 0 else self.mjx_data.ctrl,
            )

        for rb in self.robots:
            rb.receiveState()

    def add_object_rt(self, new_obj: MjXmlLoadable):
        # For all freezable objects, store internal state as initial state
        for obj in itertools.chain(self.obj_repo.get_obj_list(), self.robots):
            if isinstance(obj, MjFreezable):
                obj.freeze(self.model, self.data)

        # Register new object
        self.obj_repo.add_object(new_obj)
        self.obj_repo.register_obj_id(new_obj, len(self.obj_repo.get_obj_list()))
        if isinstance(new_obj, IntelligentSimObject):
            new_obj.register_sim((self.model, self.data), self.sim_name)

        # Rebuild scene
        self.mj_scene_parser = MjSceneParser()
        self.model, self.data = self.mj_scene_parser.create_scene(
            self.robots, self.surrounding, self.obj_repo.get_obj_list(), self.dt
        )

        reset_render_singleton()
        if self.render_mode == self.RenderMode.HUMAN:
            self.viewer.reinit(self.model, self.data)  # Renderer

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = np.zeros(self.data.qvel.shape)
        self.load_robot_to_scene()

        for obj in itertools.chain(self.obj_repo.get_obj_list(), self.robots):
            if isinstance(obj, MjFreezable):
                obj.unfreeze(self.model, self.data)
        self.rt_obj.append(new_obj)

    def obj_reset(self):
        for obj in self.rt_obj:
            self.obj_repo.remove_object(obj)

        for obj in itertools.chain(self.obj_repo.get_obj_list(), self.robots):
            if isinstance(obj, MjFreezable):
                obj.freeze(self.model, self.data)

        # Rebuild scene
        self.mj_scene_parser = MjSceneParser()
        self.model, self.data = self.mj_scene_parser.create_scene(
            self.robots, self.surrounding, self.obj_repo.get_obj_list(), self.dt
        )
        if self.render_mode == self.RenderMode.HUMAN:
            self.viewer.reinit(self.model, self.data)  # Renderer

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = np.zeros(self.data.qvel.shape)
        self.load_robot_to_scene()

        for obj in itertools.chain(self.obj_repo.get_obj_list(), self.robots):
            if isinstance(obj, MjFreezable):
                obj.unfreeze(self.model, self.data)
        self.rt_objects = []

    def start_recording(self, nframes):
        if self.render_mode == self.RenderMode.HUMAN:
            self.viewer.start_recording(nframes=nframes)

    def _setup_objects(self, sim_objs: List[SimObject]):
        for i, obj in enumerate(sim_objs):
            self.obj_repo.register_obj_id(obj, i)

            if isinstance(obj, IntelligentSimObject):
                obj.register_sim((self.model, self.data), self.sim_name)

    def _rt_add_object(self, sim_obj: SimObject):
        raise RuntimeError(
            "Adding objects in MuJoCo only possible prior to scene setup."
        )

    def _get_obj_seg_id(self, obj_name: str):
        """
        Returns the ID of an Object based on an obj_name
        This ID is the one used in the Segmentation Image retrievable through get_segmentation
        :param obj_name
        """
        return mj_name2id(self.model, 1, obj_name)

    def _get_obj_pos(self, poi, sim_obj: SimObject):
        pos, quat = self._get_obj_pos_and_quat(poi=poi, sim_obj=sim_obj)
        return pos

    def _get_obj_quat(self, poi, sim_obj: SimObject):
        pos, quat = self._get_obj_pos_and_quat(poi=poi, sim_obj=sim_obj)
        return quat

    def _get_obj_pos_and_quat(self, poi, sim_obj: SimObject):
        body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, sim_obj.name)
        body_jnt_addr = self.model.body_jntadr[body_id]
        qposadr = self.model.jnt_qposadr[body_jnt_addr]

        if body_jnt_addr == -1:
            # Object without joint
            pos = self.model.body_pos[body_id]
            quat = self.model.body_quat[body_id]
        else:
            # Object with joint
            pos = self.data.qpos[qposadr : qposadr + 3]
            quat = self.data.qpos[qposadr + 3 : qposadr + 7]

        return pos.copy(), quat.copy()

    def _set_obj_pos(self, new_pos, sim_obj: SimObject):
        self._set_obj_pos_and_quat(new_pos, None, sim_obj=sim_obj)

    def _set_obj_quat(self, new_quat, sim_obj: SimObject):
        self._set_obj_pos_and_quat(None, new_quat, sim_obj=sim_obj)

    def _set_obj_pos_and_quat(self, new_pos, new_quat, sim_obj: SimObject):
        if new_pos is None and new_quat is None:
            logging.getLogger(__name__).warning(
                "Expected at least either a new position or quaternion for set_obj_pos_and_quat"
            )
            return

        body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, sim_obj.name)
        body_jnt_addr = self.model.body_jntadr[body_id]
        qposadr = self.model.jnt_qposadr[body_jnt_addr]

        if new_pos is not None:
            assert len(new_pos) == 3, print(
                f"Expected a positions list of 3 values, got {len(new_pos)}"
            )

            if body_jnt_addr == -1:
                # Static object
                self.model.body_pos[body_id] = new_pos
            else:
                # Object with joint
                self.data.qpos[qposadr : qposadr + 3] = new_pos

        if new_quat is not None:
            assert len(new_quat) == 4, print(
                f"Expected a quaternions list of 4 values, got {len(new_quat)}"
            )
            if body_jnt_addr == -1:
                # Static object
                self.model.body_quat[body_id] = new_quat
            else:
                # Object with joint
                self.data.qpos[qposadr + 3 : qposadr + 7] = new_quat

    def _remove_object(self, sim_obj: SimObject):
        raise RuntimeError(
            "Removing objects in MuJoCo only possible prior to scene setup."
        )

    def set_state_mjx(self, qpos, qvel):
        from mujoco import mjx
        import jax.numpy as jnp

        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
        )
        self.mjx_data = mjx.forward(self.mjx_model, self.mjx_data)
    
    def set_state(self, qpos, qvel):
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)
    
    def _sync_cpu_to_mjx(self):
        """Copy current CPU mujoco data into mjx_data."""
        print("Syncing CPU mujoco data into MJX data...")
        if not (self.use_mjx and (mjx is not None) and (self.mjx_model is not None)):
            return
        if self.mjx_data is None:
            self.mjx_data = mjx.make_data(self.mjx_model)

        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.asarray(self.data.qpos),
            qvel=jnp.asarray(self.data.qvel),
            act=jnp.asarray(self.data.act) if self.model.na > 0 else self.mjx_data.act,
            ctrl=jnp.asarray(self.data.ctrl) if self.model.nu > 0 else self.mjx_data.ctrl,
        )

    def _sync_mjx_to_cpu(self, do_forward=True):
        """Copy mjx_data back into CPU mujoco data (for rendering / receiveState / legacy code)."""
        print("Syncing MJX data back to CPU mujoco data...")
        if not (self.use_mjx and (self.mjx_data is not None)):
            return

        self.data.qpos[:] = np.asarray(self.mjx_data.qpos)
        self.data.qvel[:] = np.asarray(self.mjx_data.qvel)
        if self.model.na > 0:
            self.data.act[:] = np.asarray(self.mjx_data.act)
        if self.model.nu > 0:
            self.data.ctrl[:] = np.asarray(self.mjx_data.ctrl)

        if do_forward:
            mujoco.mj_forward(self.model, self.data)