import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch

def visualize_motion(save_prefix, save_folder, outstate,infstate,label="AirIO"):
    ### visualize gt&netoutput velocity, 2d trajectory. 
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    v_gt_x, v_gt_y, v_gt_z       = torch.split(outstate['vel_gt'][0][::50,:].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate['net_vel'][0][::50,:].cpu(), 1, dim=1)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2) 

    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1]) 
   
    #visualize traj 
    ax1.plot(airTraj_x, airTraj_y, label=label)
    ax1.plot(gt_x     , gt_y     , label="Ground Truth")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    
    #visualize vel
    ax2.plot(airVel_x,label=label)
    ax2.plot(v_gt_x,label="Ground Truth")
    
    ax3.plot(airVel_y,label=label)
    ax3.plot(v_gt_y,label="Ground Truth")
    
    ax4.plot(airVel_z,label=label)
    ax4.plot(v_gt_z,label="Ground Truth")
    
    ax2.set_xlabel('time')
    ax2.set_ylabel('velocity')
    ax2.legend()
    ax3.legend()
    ax4.legend()
    save_prefix += "_state.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi = 300)
    plt.close()

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot=None, save_folder=None):
    gt_euler = np.unwrap(pp.SO3(gt_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
    outstate_euler = np.unwrap(pp.SO3(out_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi

    legend_list = ["roll", "pitch","yaw"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Orientation Comparison")
    for i in range(3):
        axs[i].plot(outstate_euler[:, i], color="b", linewidth=0.9)
        axs[i].plot(gt_euler[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["raw_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)

    if inf_rot is not None:
        infstate_euler = np.unwrap(pp.SO3(inf_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
        for i in range(3):
            axs[i].plot(infstate_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                    "raw_" + legend_list[i],
                    "gt_" + legend_list[i],
                    "AirIMU_" + legend_list[i],
                ]
            )
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=300
        )
    # plt.show()
    plt.close()


def visualize_velocity(save_prefix, gtstate, outstate, refstate=None, save_folder=None):
    legend_list = ["x", "y", "z"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Velocity Comparison")
    for i in range(3):
        axs[i].plot(outstate[:, i], color="b", linewidth=0.9)
        axs[i].plot(gtstate[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["AirIO_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)
    
    if refstate is not None:
        for i in range(3):
            axs[i].plot(refstate[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                "AirIO_" + legend_list[i], 
                "gt_" + legend_list[i],
                "IOnet" + legend_list[i],
                ]
            )

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + ".png"), dpi=300
        )
    # plt.show()
    plt.close()