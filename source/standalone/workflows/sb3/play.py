# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# 版权所有。
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于播放来自 Stable-Baselines3 的 RL 代理的检查点脚本。"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse  # 导入命令行参数解析库

from omni.isaac.lab.app import AppLauncher  # 导入应用启动器

# 添加命令行参数
parser = argparse.ArgumentParser(description="播放来自 Stable-Baselines3 的 RL 代理的检查点。")
parser.add_argument("--video", action="store_true", default=False, help="在训练期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（以步数为单位）。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用 fabric 并使用 USD I/O 操作。"
)
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点的路径。")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="当没有提供检查点时，使用最后保存的模型。否则使用最佳保存的模型。",
)
# 添加 AppLauncher CLI 参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
# 如果启用了视频录制，始终启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的部分。"""

import gymnasium as gym  # 导入 Gymnasium 库
import numpy as np  # 导入 NumPy 库
import os  # 导入操作系统库
import torch  # 导入 PyTorch 库

from stable_baselines3 import PPO  # 导入 PPO 算法
from stable_baselines3.common.vec_env import VecNormalize  # 导入向量化环境标准化

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent  # 导入环境和转换函数
from omni.isaac.lab.utils.dict import print_dict  # 导入字典打印工具

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # 导入配置解析工具
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg  # 导入 Stable Baselines3 环境包装器

def main():
    """使用 Stable-Baselines 代理进行播放。"""
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # 日志记录目录
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    # 检查检查点是否有效
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"  # 使用最后一个模型
        else:
            checkpoint = "model.zip"  # 使用最佳模型
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)  # 获取检查点路径
    else:
        checkpoint_path = args_cli.checkpoint  # 使用提供的检查点路径
    log_dir = os.path.dirname(checkpoint_path)  # 日志目录

    # 后处理代理配置
    agent_cfg = process_sb3_cfg(agent_cfg)

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果 RL 算法需要，将环境转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为视频录制包装环境
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,  # 触发条件
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 在训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 记录视频
    # 为 Stable Baselines3 包装环境
    env = Sb3VecEnvWrapper(env)

    # 正常化环境（如果需要）
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
    print(f"从以下路径加载检查点: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)  # 加载检查点

    # 重置环境
    obs = env.reset()
    timestep = 0
    # 模拟环境
    while simulation_app.is_running():
        # 在推理模式下运行所有内容
        with torch.inference_mode():
            # 代理执行
            actions, _ = agent.predict(obs, deterministic=True)  # 预测动作
            # 环境执行
            obs, _, _, _ = env.step(actions)  # 执行动作
        if args_cli.video:
            timestep += 1
            # 录制一个视频后退出播放循环
            if timestep == args_cli.video_length:
                break

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
