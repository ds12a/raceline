from track_import.track import Track
from mesh_refinement.collocation import Collocation
from scipy.stats import gmean
import math
import numpy as np
import plotly.graph_objects as go


class MeshRefinement:
    def __init__(self, collocation: Collocation, refinement_configs: dict):
        self.colloc = collocation
        self.conf = refinement_configs
        self.dist = self.colloc.end_t - self.colloc.start_t

    def run(self):
        initial_collocation = self.conf["initial_collocation"]
        initial_mesh_points = self.conf["initial_mesh_points"]
        refinement_steps = self.conf["refinement_steps"]
        sample_res = self.conf["config"]["sampling_resolution"]

        # Generates initial set of mesh points
        self.t = np.linspace(self.colloc.start_t, self.colloc.end_t, initial_mesh_points)
        self.N = np.array(
            [initial_collocation] * (initial_mesh_points - 1)
        )  # Collocation points per interval

        # Runs iter of collocation on initial points
        candidate = self.colloc.iteration(self.t, self.N)
        self.plot_candidate(candidate, "initial")

        # Samples finely for cost function error
        best_eval = candidate
        best_iter = 0
        sample_t = np.linspace(
            self.colloc.start_t,
            self.colloc.end_t,
            math.ceil(self.dist / sample_res),
        )
        _, best_cost = self.colloc.sample_cost(candidate, sample_t)
        print(f"Sampled error: {best_cost:e}")

        # Iteratively runs mesh refinement
        for i in range(refinement_steps):
            print(f"Refinement step {i + 1}/{refinement_steps}")
            self.N, self.t = self.mesh_iteration(candidate)
            print(
                f"Fitting with {len(self.N)} segments with a segment maximum of {max(self.N)} collocation points and total sum of {self.N.sum()} collocation points"
            )
            candidate = self.colloc.iteration(self.t, self.N, best_eval)
            _, new_cost = self.colloc.sample_cost(candidate, sample_t)

            print(f"Sampled error: {new_cost:e}")

            if new_cost < best_cost:
                best_eval = candidate
                best_iter = i + 1
                best_cost = new_cost

            
            self.plot_candidate(candidate, i)
            

        # track.ccw = ccw

        print(f"Fitting finished. Chose iteration {best_iter} with cost evaluation {best_cost}.")
        return best_eval

    def mesh_iteration(self, candidate):
        resolution = self.conf["config"]["sampling_resolution"]
        variation_thres = self.conf["config"]["variation_threshold"]
        divides = self.conf["config"]["h_divisions"]
        degree_increase = self.conf["config"]["p_degree_increase"]

        print("Beginning Mesh Refinement...")

        interval_starts = self.t[:-1]  # Start t of each interval
        new_t, new_N = [self.t.min()], []
        interval_costs = []
        samples = []
        geo_mean_cost = 0

        deg_counter = 0
        div_counter = 0
        skip_counter = 0

        total_samples = 0

        for i, start in enumerate(interval_starts):
            end = self.t[i + 1]

            # Sample t, remove first so it cannot be added multiple times
            # assert end_t != start_t
            sample_t = np.linspace(start, end, math.ceil((end - start) / resolution))[1:]
            samples.append(sample_t)

            # Compute costs across interval i at the end of each t
            costs, _ = self.colloc.sample_cost(candidate, sample_t)

            geo_mean_cost += np.log(costs).sum()
            total_samples += len(costs)
            interval_costs.append(costs)

        # TODO this is an approximation, check if it is satisfactory
        geo_mean_cost = np.exp(geo_mean_cost / total_samples)

        for i, start in enumerate(interval_starts):
            end = self.t[i + 1]
            sample_t = samples[i]

            costs = interval_costs[i]

            stdev = costs.std()
            mean = gmean(costs)

            if mean < geo_mean_cost:
                new_N.append(self.N[i])
                new_t.append(end)
                skip_counter += 1
                continue

            total = np.sum(costs)

            if stdev / mean > variation_thres:
                # Divide
                div_counter += 1
                cumulative = 0
                initial_points = max(
                    math.ceil(self.N[i] / (divides + 1)), self.conf["config"]["h_min_collocation"]
                )

                for j, c in enumerate(costs):
                    cumulative += c
                    if cumulative > total / (divides + 1):
                        cumulative = 0

                        new_N.append(initial_points)
                        new_t.append(sample_t[j])

                if abs(end - new_t[-1]) > 1e-7:
                    new_N.append(initial_points)
                    new_t.append(end)

            else:
                # Increase degree
                deg_counter += 1
                new_N.append(self.N[i] + degree_increase)

                new_t.append(end)

        print(f"Degree increased: {deg_counter}\tDivided: {div_counter}\tSkipped: {skip_counter}")
        assert len(new_N) + 1 == len(new_t)
        return np.array(new_N), np.array(new_t)

    def plot_candidate(self, candidate, i):
        fig = go.Figure()

        fig.add_traces(
            [
                self.colloc.track.plot_raceline_colloc(candidate),
                *self.colloc.track.plot_car_bounds(candidate,  self.colloc.vehicle.prop.g_t),
                self.colloc.track.plot_ribbon()
            ]
        )
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            legend=dict(
                orientation="h",
            ),
            title=f"iteration {i}"
        )
        fig.show()