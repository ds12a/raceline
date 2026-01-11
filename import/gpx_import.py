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
from casadi import *  # type: ignore

matplotlib.use("Qt5Agg")

FILE = "Track.gpx"
RESOLUTION = 2.0  # meters

w_c = 1e-3
w_l = 1e-3
w_r = 1e-3
w_theta = 3e3
w_mu = 1e9
w_phi = 1e9
w_nl = 1e-2
w_nr = 1e-2


def interpolate(trajectory: np.ndarray, spacing: float = 0.1) -> np.ndarray:
    """
    Fits cubic splines to the given trajectory and returns a new trajectory with
    evenly spaced points sampled from the splines. We approximate the total distance
    of the calculated splines and map the desired distances of each point to the
    default parameterization produce by scipy.
    """
    if len(trajectory) < 2:
        return trajectory

    spline, u = splprep(trajectory, s=20, k=2, per=True)

    # make really smooth for accurate distance
    u_fine = np.linspace(u.min(), u.max(), 100000)  # TODO find good number
    x_fine, y_fine, z_fine = splev(u_fine, spline)

    # calculate distance from start for each point
    dist = np.cumsum(
        np.sqrt(np.diff(x_fine) ** 2 + np.diff(y_fine) ** 2 + np.diff(z_fine) ** 2)
    )
    dist = np.insert(dist, 0, 0)

    samples = int(dist[-1] / spacing)
    target_dist = np.linspace(0, samples * spacing, samples + 1)

    u_spaced = np.interp(target_dist, dist, u_fine)  # map distance to u parameter
    # u_spaced = np.append(u_spaced, u[-1])  # make sure end waypoint is preserved

    sampled = splev(u_spaced, spline)

    return (
        dist,
        sampled,
    )  # np.column_stack([*sampled, np.zeros(sampled[0].shape[0])]) # type: ignore


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


center_nn = KDTree(np.transpose(track[1]))
_, c_nearest = center_nn.query(np.transpose(track[0]))
for i, c in enumerate(c_nearest):  # type: ignore

    # Loop over each axis
    for j in range(3):
        track[2][j].append((track[0][j][i] + track[1][j][c]) / 2.0)  # type: ignore

track[2] = np.asarray(track[2])

interpolated_track = []

for i, t in enumerate(track):
    dist, sampled = interpolate(t, 0.1 if i < 2 else RESOLUTION)  # type: ignore

    interpolated_track.append(sampled)
    interpolated_track[i] = np.asarray(interpolated_track[i])


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


opti = ca.Opti()
l = len(s_track[0][0])

# Defining state variables
x = opti.variable(l, 1)  # NOTE: dx/ds = cos(theta)cos(mu)
y = opti.variable(l, 1)  # NOTE: dy/ds = sin(theta)cos(mu)
z = opti.variable(l, 1)  # NOTE: dz/ds = -sin(mu)
theta = opti.variable(l, 1)
mu = opti.variable(l, 1)
phi = opti.variable(l, 1)
d_theta = opti.variable(l, 1)
d_mu = opti.variable(l, 1)
d_phi = opti.variable(l, 1)
n_l = opti.variable(l, 1)
n_r = opti.variable(l, 1)

chi = ca.horzcat(x, y, z, theta, mu, phi, d_theta, d_mu, d_phi, n_r, n_l)

# Defining control variables
d2_theta = opti.variable(l, 1)
d2_mu = opti.variable(l, 1)
d2_phi = opti.variable(l, 1)
d_n_l = opti.variable(l, 1)
d_n_r = opti.variable(l, 1)

nu = ca.horzcat(d2_theta, d2_mu, d2_phi, d_n_l, d_n_r)



# Defining loss function variables
# NOTE: While in the paper, it passes in state and control vectors explicitly, we choose to
# individually pass functions relevant arguments.


# 1D Placeholder variables for tracking error
_x = opti.variable(1, 1)
_y = opti.variable(1, 1)
_z = opti.variable(1, 1)
_theta = opti.variable(1, 1)
_mu = opti.variable(1, 1)
_phi = opti.variable(1, 1)
_n_l = opti.variable(1, 1)
_n_r = opti.variable(1, 1)
_d_theta = opti.variable(1, 1)
_d_mu = opti.variable(1, 1)
_d_phi = opti.variable(1, 1)
_d2_theta = opti.variable(1, 1)
_d2_mu = opti.variable(1, 1)
_d2_phi = opti.variable(1, 1)
_d_n_l = opti.variable(1, 1)
_d_n_r = opti.variable(1, 1)


e = ca.Function("e", [_x, _y, _z, _theta, _mu, _phi, _n_l, _n_r])


X = ca.horzcat(x, y, z)
N = ca.horzcat(
    cos(theta) * sin(mu) * sin(phi) - sin(theta) * cos(phi),
    sin(theta) * sin(mu) * sin(theta) + cos(theta) * cos(phi),
    cos(mu) * sin(phi),
)
B_l = X + N * ca.horzcat(n_l, n_l, n_l)
B_r = X + N * ca.horzcat(n_r, n_r, n_r)

r_c = w_theta * d2_theta**2 + w_mu * d2_mu**2 + w_phi * d2_phi**2
r_w = ca.Function("r_w", [opti.variable()], [w_nl * d_n_l**2 + w_nr * d_n_r**2])

# Function for the derivative of the state vector
f = ca.Function(
    "f",
    [theta_ph, mu_ph, d_theta_ph, d_mu_ph, d_phi_ph, d2_theta_ph, d2_mu_ph, d2_phi_ph, d_n_l_ph, d_n_r_ph],
    [
        ca.horzcat(
            cos(theta_ph) * cos(mu_ph),
            sin(theta_ph) * cos(mu_ph),
            -sin(mu),
            d_theta,
            d_mu,
            d_phi,
            d2_theta,
            d2_mu,
            d2_phi,
            d_n_l,
            d_n_r,
        )
    ],
)

for i in range(l):
    opti.subject_to(-ca.pi / 2 < mu[i] < ca.pi)
    # Updating for the timestep

    opti.subject_to(chi[(i + 1) % l] == f * RESOLUTION + chi[i])

    e[i]

    # Variables and formulae for loss function calculation


l = e + n_l + n_r


# Tracking error e(s, chi, nu)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")


# for t in interpolated_track:
#     ax.plot(*t)

# for t in track:
#     ax.scatter(*t)

for t in s_track:
    ax.scatter(*t)

ax.set_aspect("equal", adjustable="box")

plt.show()
