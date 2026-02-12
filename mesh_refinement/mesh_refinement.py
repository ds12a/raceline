from track_import.track import Track
from collocation import Collocation
import numpy as np


class MeshRefinement:
    def __init__(self, collocation: Collocation, refinement_configs: dict):
        self.colloc = collocation
        self.conf = refinement_configs

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
        track = self.colloc.iteration(t, N)

        # Samples finely for cost function error
        best_eval = track
        best_iter = 0
        sample_t = np.linspace(
            self.colloc.start_t,
            self.colloc.end_t,
            math.ceil(max_dist / sample_res),
        )
        best_cost, _ = self.colloc.sample_cost(track, sample_t)
        print(f"Sampled error: {best_cost:e}")

        # Iteratively runs mesh refinement
        for i in range(refinement_steps):
            print(f"Refinement step {i + 1}/{refinement_steps}")
            N, t = self.mesh_iteration()
            print(
                f"Fitting with {len(N)} segments with a segment maximum of {max(N)} collocation points and total sum of {N.sum()} collocation points"
            )
            track = self.colloc.iteration(t, N)
            new_cost, _ = cost_fn.sample_cost(track, sample_t)

            print(f"Sampled error: {new_cost:e}")

            if new_cost < best_cost:
                best_eval = track
                best_iter = i + 1
                best_cost = new_cost

        track.ccw = ccw  # TODO what is this

        print(f"Fitting finished. Chose iteration {best_iter} with cost evaluation {best_cost}.")
        return best_eval

    def mesh_iteration(self):
        resolution = self.conf["sampling_resolution"]
        variation_thres = self.conf["variation_threshold"]
        divides = self.conf["h_divisions"]
        degree_increase = self.conf["p_degree_increase"]

        print("Beginning Mesh Refinement...")

        interval_starts = self.t[:-1]  # Start t of each interval
        new_t, new_N = [t.min()], []
        interval_costs = []
        samples = []
        geo_mean_cost = 0

        deg_counter = 0
        div_counter = 0
        skip_counter = 0

        for i, start_t in enumerate(interval_starts):
            end_t = self.t[i + 1]

            # Sample t, remove first so it cannot be added multiple times
            # assert end_t != start_t
            sample_t = np.linspace(start_t, end_t, math.ceil((end_t - start_t) / resolution))[1:]
            samples.append(sample_t)

            # Compute costs across interval i at the end of each t
            _, costs = cost_fn.sample_cost(track, sample_t)

            geo_mean_cost += np.log(costs).sum()

            interval_costs.append(costs)

        # TODO this is an approximation, check if it is satisfactory
        geo_mean_cost = np.exp(geo_mean_cost / ((track.t[-1] - track.t[0]) / resolution))

        for i, start_t in enumerate(interval_starts):
            end_t = track.t[i + 1]
            sample_t = samples[i]

            costs = interval_costs[i]

            stdev = costs.std()
            mean = gmean(costs)

            if mean < geo_mean_cost:
                new_N.append(N[i])
                new_t.append(end_t)
                skip_counter += 1
                continue

            total = np.sum(costs)

            if stdev / mean > variation_thres:
                # Divide
                div_counter += 1
                cumulative = 0
                initial_points = max(
                    math.ceil(N[i] / (divides + 1)), self.conf["h_min_collocation"]
                )

                for j, c in enumerate(costs):
                    cumulative += c
                    if cumulative > total / (divides + 1):
                        cumulative = 0

                        new_N.append(initial_points)
                        new_t.append(sample_t[j])

                if abs(end_t - new_t[-1]) > 1e-7:
                    new_N.append(initial_points)
                    new_t.append(end_t)

            else:
                # Increase degree
                deg_counter += 1
                new_N.append(N[i] + degree_increase)

                new_t.append(end_t)

        print(f"Degree increased: {deg_counter}\tDivided: {div_counter}\tSkipped: {skip_counter}")
        assert len(new_N) + 1 == len(new_t)
        return np.array(new_N), np.array(new_t)
