## imports
import os
import warnings
import random
import torch
import sys 
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data

sys.path.append(os.path.abspath("/home/mcamp/acorn"))

from acorn.utils import (
    load_datafiles_in_dir,
    run_data_tests,
    handle_weighting,
    handle_hard_cuts,
    handle_hard_node_cuts,
    remap_from_mask,
    handle_edge_features,
    get_optimizers,
    get_condition_lambda,
)

import yaml

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import aggr

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage


### import only if using quantum network

import pennylane as qml

import matplotlib as plt
import time 
import yappi


### QNN Architecture

n_qubits=4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs,weight1,weight2):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits),rotation='Y')

    for i in range(n_qubits):
        qml.RY(weight1[i],i)

    for i in range(n_qubits-1):
        qml.CZ((n_qubits-2-i%n_qubits, n_qubits-1-i%n_qubits))
    qml.CZ((0,n_qubits-1))
    for i in range(n_qubits):
        qml.RY(weight2[i],i)
    
    return tuple(qml.expval(qml.Z(i)) for i in range(n_qubits))

### Acorn function to create GNN layers
### I did not change anything besides adding an optional QNN at the end

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,
    batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    qnn = False
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False)
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    
    # Final layer
    # Either finishes with a single linear layer
    # or finishes with a QNN and linear layer
    
    if not qnn:
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

    ### Uncomment if using quantum network
    else: 
        weight_shapes = {"weight1": n_qubits,"weight2":n_qubits}
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        layers.append(nn.Linear(sizes[-2], n_qubits))
        layers.append(qlayer)
        layers.append(nn.Linear(n_qubits,sizes[-1]))
                      
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False)
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)



## Reads from directory and creates graph data
## I edited apply_hard_cuts to cut on both nodes and edges if node_cut=True
## Acorn only cuts on edges

class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        preprocess=True,
        node_cut = False
    ):
        if hparams is None:
            hparams = {}
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        self.preprocess = preprocess
        self.node_cut = node_cut

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        if not self.preprocess:
            return event
        event = self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        # print(event)
        if self.hparams.get("undirected"):
            event = self.to_undirected(event)
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)
        event = self.add_edge_features(event)
        event = self.scale_features(event)
        return event

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if (
            self.hparams is not None
            and "hard_cuts" in self.hparams.keys()
            and self.hparams["hard_cuts"]
        ):
            assert isinstance(
                self.hparams["hard_cuts"], dict
            ), "Hard cuts must be a dictionary"
            if self.node_cut:
               handle_hard_node_cuts(event, self.hparams["hard_cuts"])
            handle_hard_cuts(event, self.hparams["hard_cuts"])
            

        return event

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.y.shape[0]} truth labels"
        )

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(
                self.hparams["weighting"][0], dict
            ), "Weighting must be a list of dictionaries"
            event.weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.weights = torch.ones_like(event.y, dtype=torch.float32)

        return event

    def handle_edge_list(self, event):
        if (
            "input_cut" in self.hparams.keys()
            and self.hparams["input_cut"]
            and "scores" in event.keys()
        ):
            # Apply a score cut to the event
            self.apply_score_cut(event, self.hparams["input_cut"])

        # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
        #     # Flip event.edge_index and concat together
        #     self.to_undirected(event)
        return event

    def to_undirected(self, event):
        """
        Add the reverse of the edge_index to the event. This then requires all edge features to be duplicated.
        Additionally, the truth map must be duplicated.
        """

        num_edges = event.edge_index.shape[1]
        # Flip event.edge_index and concat together
        event.edge_index = torch.cat(
            [event.edge_index, event.edge_index.flip(0)], dim=1
        )
        # event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
        num_track_edges = event.track_edges.shape[1]
        event.track_edges = torch.cat(
            [event.track_edges, event.track_edges.flip(0)], dim=1
        )

        # Concat all edge-like features together
        for key in event.keys():
            if key == "truth_map":
                continue
            if isinstance(event[key], torch.Tensor) and (
                (event[key].shape[0] == num_edges)
            ):
                event[key] = torch.cat([event[key], event[key]], dim=0)
                # event[key] = torch.zeros_like(event.edge_index[0], dtype=event[key].dtype).scatter(0, unique_edge_indices, event[key])

            # Concat track-like features for evaluation
            elif isinstance(event[key], torch.Tensor) and (
                (event[key].shape[0] == num_track_edges)
            ):
                event[key] = torch.cat([event[key], event[key]], dim=0)

        # handle truth_map separately
        truth_map = event.truth_map.clone()
        truth_map[truth_map >= 0] = truth_map[truth_map >= 0] + num_edges
        event.truth_map = torch.cat([event.truth_map, truth_map], dim=0)
        return event

    def add_edge_features(self, event):
        if "edge_features" in self.hparams.keys():
            assert isinstance(
                self.hparams["edge_features"], list
            ), "Edge features must be a list of strings"
            handle_edge_features(event, self.hparams["edge_features"])
        return event

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys(), f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

        return event

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys(), f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]
        return event

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        num_edges = event.edge_index.shape[1]
        for key in event.keys():
            if (
                isinstance(event[key], torch.Tensor)
                and event[key].shape
                and (
                    event[key].shape[0] == num_edges
                    or event[key].shape[-1] == num_edges
                )
            ):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)
        return event

    def get_y_node(self, event):
        y_node = torch.zeros(event.z.size(0))
        y_node[event.track_edges.view(-1)] = 1
        event.y_node = y_node
        return event


## Commonframework GNN with QNN added
class InteractionGNN(EdgeClassifierStage):

    """
    An interaction network class
    """

    def __init__(self, hparams, qnn=False):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default

        self.setup_aggregation()
        

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )


        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            qnn = qnn
            )

        # The node network computes new node features
        
        self.node_network = make_mlp(
            self.network_input_size,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            qnn=qnn
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        self.save_hyperparameters(hparams)

    def message_step(self, x, start, end, e, i):
        # Compute new node features
        edge_messages = torch.cat(
            [
                self.aggregation(e, end, dim_size=x.shape[0]),
                self.aggregation(e, start, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)
     
        x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        
        e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        scores = self.output_edge_classifier(classifier_inputs).squeeze(-1)

        if (
            self.hparams.get("undirected")
            and self.hparams.get("dataset_class") != "HeteroGraphDataset"
        ):
            scores = torch.mean(scores.view(2, -1), dim=0)

        return scores

    def forward(self, batch, **kwargs):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        
        start, end = batch.edge_index
        if "undirected" in self.hparams and self.hparams["undirected"]:
            start, end = torch.cat([start, end]), torch.cat([end, start])

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder,x,use_reentrant = False)
        e = checkpoint(self.edge_encoder,torch.cat([x[start], x[end]], dim=1),use_reentrant=False)
        

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x, e = checkpoint(self.message_step,x, start, end, e, i,use_reentrant=False)

        return self.output_step(x, start, end, e)

    def setup_aggregation(self):
        if "aggregation" not in self.hparams:
            self.hparams["aggregation"] = ["sum"]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], str):
            self.hparams["aggregation"] = [self.hparams["aggregation"]]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], list):
            self.network_input_size = (1 + 2 * len(self.hparams["aggregation"])) * (
                self.hparams["hidden"]
            )
        else:
            raise ValueError("Unknown aggregation type")

        try:
            self.aggregation = aggr.MultiAggregation(
                self.hparams["aggregation"], mode="cat"
            )
        except ValueError:
            raise ValueError(
                "Unknown aggregation type. Did you know that the latest version of"
                " GNN4ITk accepts any list of aggregations? E.g. [sum, mean], [max,"
                " min, std], etc."
            )


import numpy as np
def loss_function(output, batch, balance="proportional"):
    """
    Applies the loss function to the output of the model and the truth labels.
    To balance the positive and negative contribution, simply take the means of each separately.
    Any further fine tuning to the balance of true target, true background and fake can be handled
    with the `weighting` config option.
    """

    assert hasattr(batch, "y"), (
        "The batch does not have a truth label. Please ensure the batch has a `y`"
        " attribute."
    )
    assert hasattr(batch, "weights"), (
        "The batch does not have a weighting label. Please ensure the batch"
        " weighting is handled in preprocessing."
    )

    if balance not in ["equal", "proportional"]:
        warnings.warn(
            f"{balance} is not a proper choice for the loss balance. Use either 'equal' or 'proportional'. Automatically switching to 'proportional' instead."
        )
        balance = "proportional"

    negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0)

    negative_loss = F.binary_cross_entropy_with_logits(
        output[negative_mask],
        torch.zeros_like(output[negative_mask]),
        weight=batch.weights[negative_mask].abs(),
        reduction="sum",
    )

    positive_mask = (batch.y == 1) & (batch.weights > 0)
    positive_loss = F.binary_cross_entropy_with_logits(
        output[positive_mask],
        torch.ones_like(output[positive_mask]),
        weight=batch.weights[positive_mask].abs(),
        reduction="sum",
    )

    if balance == "proportional":
        n = positive_mask.sum() + negative_mask.sum()
        return (
            (positive_loss + negative_loss) / n,
            positive_loss.detach() / n,
            negative_loss.detach() / n,
        )
    else:
        n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
        n = n_pos + n_neg
        return (
            positive_loss / n_pos + negative_loss / n_neg,
            positive_loss.detach() / n,
            negative_loss.detach() / n,
        )

