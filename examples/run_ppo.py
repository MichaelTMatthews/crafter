import argparse
import time

import crafter
import stable_baselines3

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=1e4)
args = parser.parse_args()

env = crafter.Env()
# env = crafter.Recorder(
#     env, args.outdir,
#     save_stats=True,
#     save_episode=False,
#     save_video=False,
# )

t0 = time.time()
model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)

print(f"Time: {time.time() - t0:.2f}s")

# t0 = time.time()
#
# for _ in range(int(1e4)):
#     action = np.random.randint(17)
#     obs, reward, done, info = env.step(action)
#     if done:
#         env.reset()
#
# print(f"Time: {time.time() - t0:.2f}s")