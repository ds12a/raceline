import argparse
import yaml
from track_fitting import fit_track
from gpx_parsing import read_gpx_splines
from scipy.interpolate import splev, splprep
import plotly.graph_objects as go
import numpy as np

RESOLUTION = 15

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gpx_source", required=True, type=str, help="Source path to track gpx file."
)
parser.add_argument(
    "--track_destination",
    default="track.json",
    type=str,
    help="Destination path of fitted track.",
)
parser.add_argument(
    "--config", default="config/default.yaml", type=str, help="Path to config file."
)
parser.add_argument(
    "--plot", default=False, action="store_true", help="Toggles on plotting."
)


args = parser.parse_args()


with open(args.config, "r") as file:
    config_data = yaml.safe_load(file)

s_track = [0, 0, 0]

original_track, (
    max_dist,
    spline_l,
    spline_r,
    spline_c,
    s_track[0],
    s_track[1],
    s_track[2],
) = read_gpx_splines(args.gpx_source)

# Checking Clockwise or Counterclockwise via Greene's Theorem
sample_space = np.linspace(0, max_dist, 1_000)
sample = splev(sample_space, spline_c)
area = np.sum(
    np.array([sample[0][i] * sample[1][i + 1] - sample[0][i + 1] * sample[1][i]])
    for i in range(999)
)  # Hacky "trapezoidal" approximation

ccw = False
# Is counterclockwise, reverse parameterization
if area > 0:
    print("Re-parameterizing for CCW track...")
    ccw = True
    fine_sample = np.linspace(max_dist, 0, int(max_dist // RESOLUTION))
    sample_c = splev(fine_sample, spline_c)
    sample_l = splev(fine_sample, spline_l)
    sample_r = splev(fine_sample, spline_r)

    fine_sample = fine_sample[::-1]     # Reverse parameterization
    spline_c, _ = splprep(sample_c, u=fine_sample)
    spline_l, _ = splprep(sample_l, u=fine_sample)
    spline_r, _ = splprep(sample_r, u=fine_sample)

track = fit_track(spline_c, spline_l, spline_r, max_dist, config_data["track_fitting"], ccw=ccw)
track.save(args.track_destination)

if args.plot:
    plots = []

    plots.append(
        go.Scatter3d(
            x=original_track[2][0],
            y=original_track[2][1],
            z=original_track[2][2],
            name="original center",
            mode="lines",
        )
    )

    plots.append(
        go.Scatter3d(
            x=original_track[0][0],
            y=original_track[0][1],
            z=original_track[0][2],
            name="original left",
            mode="lines",
        )
    )
    plots.append(
        go.Scatter3d(
            x=original_track[1][0],
            y=original_track[1][1],
            z=original_track[1][2],
            name="original right",
            mode="lines",
        )
    )

    fig = go.Figure()

    fine_plot, q_fine = track.plot_uniform(1)
    collocation_plot, q_collocation = track.plot_collocation()

    for i in fine_plot:
        fig.add_trace(i)

    for i in collocation_plot:
        fig.add_trace(i)

    for i in plots:
        fig.add_trace(i)

    fig.show()

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

    # Plot theta, mu, phi
    q_fig = go.Figure(data=[q_collocation, q_fine])
    q_fig.show()
