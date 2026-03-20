import numpy as np
future_points_drawn = 5
timestep = 0.25
points = np.zeros((future_points_drawn, 3), dtype=np.float64)
for i in range(future_points_drawn):
    points[i, :] = np.asanyarray([i, i, i])

print(points)
print(np.shape(points))