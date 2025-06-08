import numpy as np
import robosuite as suite
from deep_inference import DeepEEPolicy
import time

env = suite.make(
    env_name="NutAssemblySquare",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
)

policy = DeepEEPolicy(model_path="policy.pth")

success_count = 0
n_trials = 5  



# query the policy for the number of trials you wish to test on
for trial in range(n_trials):
    obs = env.reset()
    for _ in range(2500):
        ee_pos  = obs["robot0_eef_pos"]   
        sq_pos  = obs["SquareNut_pos"]    
        sq_quat = obs["SquareNut_quat"]   
        action  = policy.get_action(ee_pos, sq_pos, sq_quat)  
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1/1000)
        if reward == 1.0:
            success_count += 1
            break

success_rate = success_count / float(n_trials)
print("success rate over", n_trials, "trials:", success_rate)




