from atlasify import atlasify
import numpy as np

import torch

from acorn.stages.track_building.utils import rearrange_by_distance
from acorn.utils.plotting_utils import (
    get_ratio,
    plot_1d_histogram,
)
from acorn.utils import get_condition_lambda
from acorn.stages.graph_construction.models.utils import graph_intersection


import os



def apply_score_cut(event, score_cut):
    """
    Apply a score cut to the event. This is used for the evaluation stage.
    """
    passing_edges_mask = event.scores >= score_cut

    # flip edge direction if points inward
    event.edge_index = rearrange_by_distance(event, event.edge_index)
    event.track_edges = rearrange_by_distance(event, event.track_edges)

    event.graph_truth_map = graph_intersection(
        event.edge_index,
        event.track_edges,
        return_y_pred=False,
        return_y_truth=False,
        return_truth_to_pred=True,
    )
    event.truth_map = graph_intersection(
        event.edge_index[:, passing_edges_mask],
        event.track_edges,
        return_y_pred=False,
        return_truth_to_pred=True,
    )
    event.pred = passing_edges_mask


def apply_target_conditions(event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks * condition_lambda(event)

        event.target_mask = passing_tracks

def graph_scoring_efficiency(dataset, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    print("Plotting efficiency against pT and eta")
    true_positive, target_pt, target_eta = [], [], []
    pred = []
    graph_truth = []

    for event in dataset:
        event = event.to('cpu')
        
        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)
        
        # get all target true positives
        true_positive.append((event.truth_map[event.target_mask]>-1).cpu())
        # get all target pt. Length = number of target true
        target_pt.append(event.pt[event.target_mask].cpu())
        # target_eta.append(event.eta[event.target_mask])
        target_eta.append(event.eta[event.track_edges[:, event.target_mask][0]])
        # get all edges passing edge cut
        if "scores" in event.keys():
            pred.append((event.scores >= config["score_cut"]).cpu())
        else:
            pred.append(event.y.cpu())
        # get all target edges in input graphs
        graph_truth.append((event.graph_truth_map[event.target_mask] > -1))

    # concat all target pt and eta
    target_pt = torch.cat(target_pt).cpu().numpy()
    target_eta = torch.cat(target_eta).cpu().numpy()

    # get all true positive
    true_positive = torch.cat(true_positive).cpu().numpy()
    
    # get all positive
    graph_truth = torch.cat(graph_truth).cpu().numpy()

    # count number of graphs to calculate mean efficiency
    n_graphs = len(pred)

    # get all predictions
    pred = torch.cat(pred).cpu().numpy()

    # get mean graph size
    mean_graph_size = pred.sum() / n_graphs

    # get mean target efficiency
    target_efficiency = true_positive.sum() / len(target_pt)
    target_purity = true_positive.sum() / pred.sum()
    
    # get graph construction efficiency
    graph_construction_efficiency = graph_truth.mean()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min, pt_max = 1, 50
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    eta_bins = np.linspace(-4, 4)

    true_pt_hist, true_pt_bins = np.histogram(target_pt, bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[true_positive], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(target_eta, bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(target_eta[true_positive], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    filename = plot_config.get("filename", "edgewise_efficiency")

    for true_pos_hist, true_hist, bins, xlabel, logx, filename in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [true_pt_bins, true_eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [True, False],
        [f"{filename}_pt.png", f"{filename}_eta.png"],
    ):
        # Divide the two histograms to get the edgewise efficiency
        hist, err = get_ratio(true_pos_hist, true_hist)

        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            plot_config["title"],
            plot_config.get("ylim", [0.40, 1.04]),
            "Efficiency",
            logx=logx,
        )

        # Save the plot
        atlasify(
            "Internal",
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) "
            + "\n"
            r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(config["score_cut"]) + "\n"
            f"Input graph size: {pred.shape[0]/n_graphs:.2e}, Graph Construction Efficiency: {graph_construction_efficiency:.3f}"
            + "\n"
            f"Mean graph size: {mean_graph_size:.2e}, Signal Efficiency: {target_efficiency:.3f}",
        )

        fig.savefig(os.path.join(config["stage_dir"], filename))
        print(
            f'Finish plotting. Find the plot at {os.path.join(config["stage_dir"], filename)}'
        )
