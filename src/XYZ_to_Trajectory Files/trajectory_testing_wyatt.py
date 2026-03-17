import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

CSV_FILE = "camera_xyz_log.csv"
MIN_CLUSTER_POINTS = 8
TIME_GAP_MS = 200
GRAVITY = 9.81

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_FILE, header=None)
df.columns = ["time","x","y","z"]

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# =========================
# REMOVE ZERO DETECTIONS
# =========================

valid = df[(df["x"]!=0) | (df["y"]!=0) | (df["z"]!=0)]

# =========================
# FIND CLUSTERS
# =========================

t = valid["time"].values
clusters=[]
start=0

for i in range(1,len(t)):
    if t[i]-t[i-1] > TIME_GAP_MS:
        clusters.append(valid.iloc[start:i])
        start=i

clusters.append(valid.iloc[start:])
clusters=[c for c in clusters if len(c)>=MIN_CLUSTER_POINTS]

traj = clusters[0]

# =========================
# EXTRACT DATA
# =========================

t = traj["time"].values
t = (t - t[0])/1000.0

x = medfilt(traj["x"].values,5)
y = medfilt(traj["y"].values,5)
z = medfilt(traj["z"].values,5)

# =========================
# PLOTTING SETUP
# =========================

plt.figure()

plt.scatter(t,z,label="Measured",color="black")

# =========================
# INCREMENTAL FIT
# =========================

for i in range(MIN_CLUSTER_POINTS,len(t)):

    t_sub = t[:i]
    x_sub = x[:i]
    y_sub = y[:i]
    z_sub = z[:i]

    A = np.column_stack((np.ones(len(t_sub)),t_sub))

    x0,vx = np.linalg.lstsq(A,x_sub,rcond=None)[0]
    y0,vy = np.linalg.lstsq(A,y_sub,rcond=None)[0]

    z_adj = z_sub + 0.5*GRAVITY*t_sub**2
    z0,vz = np.linalg.lstsq(A,z_adj,rcond=None)[0]

    t_model = np.linspace(0,max(t)+0.5,200)
    Z = z0 + vz*t_model - 0.5*GRAVITY*t_model**2

    # how much time was used for this estimate
    time_used = t_sub[-1]

    # only label every 5th curve to avoid legend explosion
    if i % 5 == 0:
        label = f"{time_used:.2f} s data"
        plt.plot(t_model,Z,alpha=0.3,label=label)
    else:
        plt.plot(t_model,Z,alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Z (m)")
plt.title("Trajectory Estimate Convergence")

plt.show()

from mpl_toolkits.mplot3d import Axes3D

plt.figure()

ax = plt.axes(projection="3d")

# measured data
ax.scatter(x,y,z,label="Measured",s=20)

# final trajectory estimate using ALL data
A = np.column_stack((np.ones(len(t)),t))

x0,vx = np.linalg.lstsq(A,x,rcond=None)[0]
y0,vy = np.linalg.lstsq(A,y,rcond=None)[0]

z_adj = z + 0.5*GRAVITY*t**2
z0,vz = np.linalg.lstsq(A,z_adj,rcond=None)[0]

t_model = np.linspace(0,max(t)+0.5,200)

X = x0 + vx*t_model
Y = y0 + vy*t_model
Z = z0 + vz*t_model - 0.5*GRAVITY*t_model**2

ax.plot3D(X,Y,Z,label="Fitted Trajectory",linewidth=2)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

plt.show()
ax.set_box_aspect([1,1,1])


# ==============================
# INTERCEPTION
# ==============================

# z_target = 0.5

# a = -0.5*GRAVITY
# b = vz
# c = z0 - z_target

# roots = np.roots([a,b,c])
# t_hit = np.max(roots)

# x_hit = x0 + vx*t_hit
# y_hit = y0 + vy*t_hit