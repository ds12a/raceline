import os, sys

sys.path.append(os.path.dirname(__file__))

import argparse
import yaml
from track_fitting import fit_track
from gpx_parsing import read_gpx_splines
from scipy.interpolate import splev, splprep
import plotly.graph_objects as go
import numpy as np

if __name__ == "__main__":
    RESOLUTION = 15

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpx_source", required=True, type=str, help="Source path to track gpx file."
    )
    parser.add_argument(
        "--track_destination",
        default="generated/track.json",
        type=str,
        help="Destination path of fitted track.",
    )
    parser.add_argument(
        "--config", default="config/default.yaml", type=str, help="Path to config file."
    )
    parser.add_argument("--plot", default=False, action="store_true", help="Toggles on plotting.")
    parser.add_argument(
        "--solver", default="mumps", type=str, help="Solver to use (mumps, ma57, ma86, ma97, etc.)."
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)

    config_data["track_fitting"]["ipopt"]["ipopt.linear_solver"] = args.solver

    (
        original_track,
        (
            max_dist,
            spline_l,
            spline_r,
            spline_c,
            _,
            _,
            _,
        ),
        ccw,
    ) = read_gpx_splines(args.gpx_source)

    track = fit_track(spline_c, spline_l, spline_r, max_dist, config_data["track_fitting"], ccw)
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
