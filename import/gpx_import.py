from dataclasses import dataclass
import casadi as ca
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gpxpy
import gpxpy.gpx
from pyproj import Transformer
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree
from casadi import *
import plotly.express as px
import plotly.graph_objects as go


matplotlib.use("Qt5Agg")

FILE = "Track.gpx"
RESOLUTION = 10.0  # meters


def interpolate(sample: np.ndarray, spacing: float = 0.1) -> np.ndarray:
    """
    Fits polynomial splines to the given sample and returns a new array with
    evenly spaced points sampled from the splines. We approximate the total distance
    of the calculated splines and map the desired distances of each point to the
    default parameterization produce by scipy.

    Args:
        sample (np.ndarray):    A 2-dimensional array in the format [[point],[point], ... ].
                                For convenience, the method parameterizes based on arc length
                                of the first three values in each point assumed (x,y,z)
        spacing (float, optional):  The distance spacing between each of the points in the array
                                    that is returned. Defaults to 0.1.

    Returns:
        np.ndarray: A set of evenly-spaced points along the interpolated spline
    """

    # Creates the initial spline
    # u is an NDArray that contains scipy's internal parameterization for every data point in sample
    spline, u = splprep(sample, s=20, k=2, per=True)

    # Samples very finely along the spline interpolation for accurate distance parameterization
    u_fine = np.linspace(u.min(), u.max(), 1_000_000)
    fine_sample = splev(u_fine, spline)

    # Calculates Euclidian distance at each point in the fine sample
    dist = np.cumsum(
        np.sqrt(
            np.diff(fine_sample[0]) ** 2
            + np.diff(fine_sample[1]) ** 2
            + np.diff(fine_sample[2]) ** 2
        )
    )
    dist = np.insert(dist, 0, 0)  # 0 inserted for first point

    # Calculates number of samples and generates distances to samplee at with even spacing
    samples = int(dist[-1] / spacing)
    target_dist = np.linspace(0, samples * spacing, samples + 1)

    # Generates u_spaced by estimating the u parameterization based on the relationship between
    # Euclidian distance (dist) and scipy parameterization u.
    u_spaced = np.interp(target_dist, dist, u_fine)
    u_spaced = np.append(u_spaced, u[-1])  # make sure end waypoint is preserved

    # Evaluates spline
    sampled = splev(u_spaced, spline)

    # Returns Euclidean distance to each point and spline samples
    return dist, sampled



# Reads gpx file
with open(FILE) as file:
    gpx = gpxpy.parse(file)


trm = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)

# 0 - Outside/Left, 1 - Inside/Right, 2 - Center
# x, y, z:
#   1st dim - outside/inside
#   2nd dim - [[x], [y], [z]]
track: list[list | np.ndarray] = [[[], [], []], [[], [], []], [[], [], []]]

# Origin
zero_pt = gpx.tracks[0].segments[0].points[0]  # lat long
glob_zero_x, glob_zero_y = trm.transform(
    zero_pt.longitude, zero_pt.latitude
)  # cartesian

# Fills in track, generates x, y, z coordinates from (lat, long, elev)
for t in gpx.tracks:
    if t.name == "Outside":
        i = 0
    elif t.name == "Inside":
        i = 1
    else:
        raise ValueError(f"{FILE} must contain only tracks 'Outside' and 'Inside'")

    for s in t.segments:
        for p in s.points:
            x, y = trm.transform(p.longitude, p.latitude)

            # Shifts coordinates according to global zero
            track[i][0].append(x - glob_zero_x)
            track[i][1].append(y - glob_zero_y)
            track[i][2].append(p.elevation)
    track[i] = np.asarray(track[i])


# Set minimum elevation to 0
z_min = min(np.min(track[0][2]), np.min(track[1][2]))
track[0][2] -= z_min
track[1][2] -= z_min

# Query for nearest point on the right for each left point to generate the centerline
center_nn = KDTree(np.transpose(track[1]))
_, c_nearest = center_nn.query(np.transpose(track[0]))
for i, c in enumerate(c_nearest):  # type: ignore

    # Loop over each axis
    for j in range(3):
        track[2][j].append((track[0][j][i] + track[1][j][c]) / 2.0)  # type: ignore

track[2] = np.asarray(track[2])

interpolated_track = []
dists = []

for i, t in enumerate(track):
    dist, sampled = interpolate(t, 0.1 if i < 2 else RESOLUTION)  # type: ignore

    interpolated_track.append(sampled)
    interpolated_track[i] = np.asarray(interpolated_track[i])
    dists.append(dist)


out_nn = KDTree(np.transpose(interpolated_track[0][:2]))
in_nn = KDTree(np.transpose(interpolated_track[1][:2]))


_, o_nearest = out_nn.query(np.transpose(interpolated_track[2][:2]))
_, i_nearest = in_nn.query(np.transpose(interpolated_track[2][:2]))

# [out/1, in/r, center]
s_track = [
    interpolated_track[0][:, o_nearest],
    interpolated_track[1][:, i_nearest],
    interpolated_track[2],
]

spline_l, _ = splprep(s_track[0], u=dists[2])
spline_r, _ = splprep(s_track[1], u=dists[2])
spline_C, _ = splprep(s_track[2], u=dists[2])


plots = []
# for t in interpolated_track:
#     ax.plot(*t)

# for t in track:
#     ax.scatter(*t)

for t in s_track:
    plots.append(go.Scatter3d(x=t[0], y=t[1], z=t[2]))
    print("test")

fig = go.Figure(data=plots)
fig.update_layout(scene=dict(aspectmode="data"))

fig.show()
