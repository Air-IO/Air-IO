import os

import numpy as np
import pypose as pp
import torch

from utils import lookAt, qinterp, Gaussian_noise
from .dataset import Sequence
import pickle

class Euroc(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007, 
        remove_g=False,
        **kwargs
    ):
        super(Euroc, self).__init__()
        (
            self.data_root,
            self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        
        self.g_vector = torch.tensor([0, 0, gravity],dtype=torch.double)

        data_path = os.path.join(data_root, data_name)
        # load imu data
        self.load_imu(data_path)
        self.load_gt(data_path)
        # EUROC require an interpolation

        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])
        # find the index of the start and end
        idx_start_imu = np.searchsorted(self.data["time"], t_start)
        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)

        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")

        for k in ["gt_time", "pos", "quat", "velocity"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

        ## start interpotation
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )

        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["velocity"]
        )


        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])


        # when evaluation: load airimu or integrated orientation:
        self.set_orientation(rot_path, data_name, rot_type)
        
        # transform to global/body frame:
        self.update_coordinate(coordinate, mode)
        
        # remove gravity term
        self.remove_gravity(remove_g)
        
    def get_length(self):
        return self.data["time"].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(
            os.path.join(folder, "mav0/imu0/data.csv"), 
            dtype=float, 
            delimiter=",",
            encoding="utf-8-sig" # added this new line
        )
        self.data["time"] = imu_data[:, 0] / 1e9
        self.data["gyro"] = imu_data[:, 1:4]  # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:, 4:]  # acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
    def load_gt(self, folder):
        gt_data = np.loadtxt(
            os.path.join(folder, "mav0/state_groundtruth_estimate0/data.csv"),
            dtype=float,
            delimiter=",",
            encoding="utf-8-sig"
        )
        self.data["gt_time"] = gt_data[:, 0] / 1e9
        self.data["pos"] = gt_data[:, 1:4]
        self.data["quat"] = gt_data[:, 4:8]  # w, x, y, z
        self.data["velocity"] = gt_data[:, -9:-6]

    def interp_rot(self, time, opt_time, quat):
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double()
        self.data["rot_wxyz"] = quat
        rot = torch.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]
        return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):

        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def update_coordinate(self, coordinate, mode):
        """
        Updates the data (imu / velocity) based on the required mode.
        :param coordinate: The target coordinate system ('glob_coord' or 'body_coord').
        :param mode: The dataset mode, only rotating the velocity during training. 
        """
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise e

    def set_orientation(self, exp_path, data_name, rotation_type):
        """
        Sets the ground truth orientation based on the provided rotation.
        :param exp_path: Path to the pickle file containing orientation data.
        :param rotation_type: The type of rotation within the pickle file (AirIMU corrected orientation / raw imu Pre-integration).
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)

            state = loaded_data[data_name]

            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                print(f"Unsupported rotation type: {rotation_type}")
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {exp_path} was not found.")
            raise

    def remove_gravity(self,remove_g):
        if remove_g is True:
                print("gravity has been removed")
                self.data["acc"] -= self.g_vector