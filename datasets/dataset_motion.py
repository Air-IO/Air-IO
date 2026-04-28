import argparse

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory
from .dataset import Sequence, SeqeuncesDataset
import math
import pypose as pp
from datasets.dataset_utils import custom_collate

class SeqeuncesMotionDataset(SeqeuncesDataset):
    def __init__(
        self,
        data_set_config,
        mode=None,
        data_path=None,
        data_root=None,
        device="cuda:0",
    ):
        super().__init__(
        data_set_config=data_set_config,
        mode=mode,
        data_path=data_path,
        data_root=data_root,
        device=device,
        )
        print(f"******* Loading {data_set_config.mode} dataset *******")
        print(f"loaded: {data_set_config.data_list[0]['data_root']}")
        if "coordinate" in data_set_config:
            print(f"coordinate: {data_set_config.coordinate}")
        if "remove_g" in data_set_config and data_set_config.remove_g is True:
            print(f"gravity has been removed")
        if "rot_type" in data_set_config:
            if data_set_config.rot_type is None:
                print(f"using groundtruth orientation")
            elif data_set_config.rot_type.lower() == "airimu":
                print(f"Using AirIMU orientation loaded from {data_set_config.rot_path}.")
            elif data_set_config.rot_type.lower() == "integration":
                print(f"Using pre-integration orientation loaded from {data_set_config.rot_path}.")
        print(f"gravity: {data_set_config.gravity}")

    
    def load_data(self, seq, start_frame, end_frame):
        self.ts.append(seq.data["time"][start_frame:end_frame + 1])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        self.dt.append(seq.data["dt"][start_frame : end_frame + 1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame : end_frame + 1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame : end_frame + 1])
        self.gt_velo.append(seq.data["velocity"][start_frame : end_frame + 1])

    def construct_index_map(self, conf, data_root, data_name, seq_id):
        seq = self.DataClass[conf.name](
            data_root, data_name,  **self.conf
        )
        seq_len = seq.get_length() - 1
        window_size, step_size = conf.window_size, conf.step_size
        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_70':
            end_frame = np.floor(seq_len * 0.7).astype(int)
        elif self.mode == 'test_30':
            start_frame = np.floor(seq_len * 0.7).astype(int)


        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
            if self.index_map[-1][2] < _duration:
                self.index_map += [[seq_id, self.index_map[-1][2], seq_len]]
        elif self.mode == "evaluate":
            # adding the last piece for evaluation
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
        else:
            sub_index_map = [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size - step_size, step_size)
                if torch.all(seq.data["mask"][j : j + window_size])
            ]
            self.index_map += sub_index_map

        ## Loading the data from each sequence into
        self.load_data(seq, start_frame, end_frame)
        

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        data = {
            'timestamp':  self.ts[seq_id][frame_id:end_frame_id + 1],
            'dt': self.dt[seq_id][frame_id: end_frame_id+1],
            'acc':self.acc[seq_id][frame_id: end_frame_id],
            'gyro':self.gyro[seq_id][frame_id: end_frame_id],
            'rot':self.gt_ori[seq_id][frame_id: end_frame_id]
        }
        init_state = {
            'init_rot':self.gt_ori[seq_id][frame_id][None, ...],
            'init_pos':self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel':self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos':self.gt_pos[seq_id][frame_id : end_frame_id+1],
            'gt_rot':self.gt_ori[seq_id][frame_id : end_frame_id+1],
            'gt_vel':self.gt_velo[seq_id][frame_id : end_frame_id+1],
        }
        return {**data, **init_state, **label}

    def get_init_value(self):
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/BaselineEuRoC.conf",
        help="config file path, i.e., configs/Euroc.conf",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")

    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config)

    dataset = SeqeuncesMotionDataset(data_set_config=conf.train)
    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
    )

    for i, (data, init, _label) in enumerate(loader):
        for k in data:
            print(k, ":", data[k].shape)
        for k in init:
            print(k, ":", init[k].shape)
