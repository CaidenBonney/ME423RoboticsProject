import csv
import numpy as np
import pandas as pd
from Trajectory import Trajectory
read_file_path = 'sim_data_test\camera_xyz_log.csv'
predictions = []
traj = Trajectory()
with open(read_file_path, mode ='r') as read_file:    
    csvFile = csv.reader(read_file)
    for line in csvFile:
        timestamp = line[0]
        xyz = np.asarray(line[1:]).reshape(3)
        if np.equal(xyz, np.array([0,0,0])):
            continue
        traj.update_trajectory()
        t_hit = traj.estimateCatchTime()
        predictions.append([t_hit, traj.pred_pos(t_hit)])
