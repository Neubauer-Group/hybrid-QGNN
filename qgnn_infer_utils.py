## Infer Step
import torch
from acorn_pennylane import loss_function
import os
from acorn.stages.graph_construction.models.utils import graph_intersection

def shared_evaluation(model, batch):
        output = model(batch)
        print(output)
        loss = loss_function(output, batch)

        scores = torch.sigmoid(output)
        batch.scores = scores.detach()

        all_truth = batch.y.bool()
        target_truth = (batch.weights > 0) & all_truth

        return {
            "loss": loss,
            "all_truth": all_truth,
            "target_truth": target_truth,
            "output": output,
            "batch": batch,
        }


 
def save_edge_scores(event, scores, dataset, path):
    event = dataset.unscale_features(event)

   
    event.truth_map = graph_intersection(
        event.edge_index,
        event.track_edges,
        return_y_pred=False,
        return_y_truth=False,
        return_truth_to_pred=True,
    )

    
    event_id = (
        event.event_id[0] if isinstance(event.event_id, list) else event.event_id
    )
    torch.save(
        event.cpu(),
        os.path.join(path, f"event{event_id}.pyg")
    )

def predict_step(model, batch, dataloader, path ):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """

        dataset = dataloader.dataset
        event_id = (
            batch.event_id[0] if isinstance(batch.event_id, list) else batch.event_id
        )
     
        eval_dict = shared_evaluation(model,batch)
        scores = torch.sigmoid(eval_dict["output"])
        batch = eval_dict["batch"]
        save_edge_scores(batch, scores, dataset, path)

