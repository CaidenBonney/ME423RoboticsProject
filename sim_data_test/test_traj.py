import io
import csv
import numpy as np
import pandas as pd
from Trajectory import Trajectory
from TrajectoryEstimator import KalmanTrajectory
read_file_path = 'sim_data_test\camera_xyz_log_1.csv'
predictions = []
traj = Trajectory()
with open(read_file_path, mode ='r') as csv_file:    
    csv_file.readline()
    for line in csv_file:
        line = line.split(',')
        timestamp = float(line[0])
        xyz = np.asarray([float(i) for i in line[1:]],dtype=float).reshape(3)
        if np.array_equal(xyz, np.array([0,0,0],dtype=float)):
            continue
        traj.update_trajectory(timestamp, xyz)
        t_hit = traj.estimateCatchTime()
        predictions.append([t_hit, traj.pred_pos(t_hit)])
