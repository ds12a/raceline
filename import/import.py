import argparse
import yaml
from track_fitting import fit_track
from gpx_parsing import read_gpx_splines
import plotly.graph_objects as go

parser = argparse.ArgumentParser()

parser.add_argument("--gpx_source", required=True, type=str, help="Source path to track gpx file.")
parser.add_argument("--track_destination", default="track.json", type=str, help="Destination path of fitted track.")
parser.add_argument("--config", default="config/default.yaml", type=str, help="Path to config file.")
parser.add_argument("--plot",  default=False, action="store_true", help="Toggles on plotting.")


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

track = fit_track(spline_c, spline_l, spline_r, max_dist, config_data["track_fitting"])
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