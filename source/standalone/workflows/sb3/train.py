# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# 版权所有。
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于训练强化学习代理的脚本，使用 Stable Baselines3。

由于 Stable-Baselines3 不支持直接在 GPU 上使用缓冲区，
我们建议使用较少数量的环境。否则，
在 GPU 和 CPU 之间的传输将会带来显著的开销。
"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse  # 导入命令行参数解析库
import sys  # 导入系统库

from omni.isaac.lab.app import AppLauncher  # 导入应用启动器

# 添加命令行参数
parser = argparse.ArgumentParser(description="使用 Stable-Baselines3 训练 RL 代理。")
parser.add_argument("--video", action="store_true", default=False, help="在训练期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（以步数为单位）。")
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制之间的间隔（以步数为单位）。")
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--seed", type=int, default=None, help="用于环境的随机种子。")
parser.add_argument("--max_iterations", type=int, default=None, help="RL 策略训练的迭代次数。")
# 添加 AppLauncher CLI 参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli, hydra_args = parser.parse_known_args()
# 如果启用了视频录制，始终启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 清除 sys.argv 以适应 Hydra
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的部分。"""

import gymnasium as gym  # 导入 Gymnasium 库
import numpy as np  # 导入 NumPy 库
import os  # 导入操作系统库
import random  # 导入随机数库
from datetime import datetime  # 导入日期时间库

from stable_baselines3 import PPO  # 导入 PPO 算法
from stable_baselines3.common.callbacks import CheckpointCallback  # 导入检查点回调
from stable_baselines3.common.logger import configure  # 导入日志配置
from stable_baselines3.common.vec_env import VecNormalize  # 导入向量化环境标准化

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)  # 导入环境和配置
from omni.isaac.lab.utils.dict import print_dict  # 导入字典打印工具
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml  # 导入文件保存工具

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config  # 导入 Hydra 任务配置
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg  # 导入 Stable Baselines3 环境包装器

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """使用 Stable-Baselines 代理进行训练。"""
    # 如果种子为 -1，则随机生成一个种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 用非 Hydra CLI 参数覆盖配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # 设置最大训练迭代次数
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # 设置环境的随机种子
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 日志记录目录
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # 将配置保存到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 后处理代理配置
    agent_cfg = process_sb3_cfg(agent_cfg)
    # 读取关于代理训练的配置
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果 RL 算法需要，将环境转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为视频录制包装环境
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 在训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 为 Stable Baselines3 包装环境
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # 从 Stable Baselines3 创建代理
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    # 配置日志记录器
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # 代理的回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    # 训练代理
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # 保存最终模型
    agent.save(os.path.join(log_dir, "model"))

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
