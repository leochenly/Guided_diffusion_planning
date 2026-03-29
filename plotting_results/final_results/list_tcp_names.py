import mujoco
from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

KEYS = ["tcp", "hand", "ee", "tip", "grip", "endeff", "end_eff", "panda"]

def hits_for(obj_type, n, model):
    hits = []
    for i in range(n):
        nm = mujoco.mj_id2name(model, obj_type, i)
        if not nm:
            continue
        s = nm.lower()
        if any(k in s for k in KEYS):
            hits.append((i, nm))
    return hits

def main():
    env = ObstacleAvoidanceEnv(render=False)
    env.start()
    env.reset()

    m = env.scene.model

    print("\n=== BODY hits ===")
    for i, nm in hits_for(mujoco.mjtObj.mjOBJ_BODY, m.nbody, m):
        print(i, nm)

    print("\n=== SITE hits ===")
    for i, nm in hits_for(mujoco.mjtObj.mjOBJ_SITE, m.nsite, m):
        print(i, nm)

    print("\n=== JOINT hits ===")
    for i, nm in hits_for(mujoco.mjtObj.mjOBJ_JOINT, m.njnt, m):
        print(i, nm)

    print("\n=== GEOM hits ===")
    for i, nm in hits_for(mujoco.mjtObj.mjOBJ_GEOM, m.ngeom, m):
        print(i, nm)

if __name__ == "__main__":
    main()