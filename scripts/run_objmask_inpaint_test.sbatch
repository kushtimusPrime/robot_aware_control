#!/usr/bin/env bash
#SBATCH --ntasks=3
#SBATCH --cpus-per-task 2
#SBATCH --gpus=2
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=inpaint_mask_table_test

# Test the effect of robot mask w/ obj occlusion
# Try inpaint, dontcare cost functions
# For inpaint, run on obj only demonstrations
# For dontcare, run on obj only demonstrations

# Dontcare + Robot mask
python -um src.mbrl.episode_runner --wandb True --jobname dontcare_realrobotmask_table_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/table_realrobotmask_straight_push --demo_type object_only_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --robot_mask_with_obj False --gpu 0 >dontcare_realrobotmask_table_envcem.out 2>&1 &

# Inpaint + Robot mask
python -um src.mbrl.episode_runner --wandb True --jobname inpaint_realrobotmask_table_envcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/table_realrobotmask_straight_push --demo_type object_only_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --robot_mask_with_obj False --gpu 0 >inpaint_realrobotmask_table_envcem.out 2>&1 &


# baseline no-inpaint loss
python -um src.mbrl.episode_runner --wandb True --jobname noinpaint_realrobotmask_table_envcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 15000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/table_realrobotmask_straight_push --demo_type object_only_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --robot_mask_with_obj False --gpu 1 >noinpaint_realrobotmask_table_envcem.out 2>&1 &
wait
