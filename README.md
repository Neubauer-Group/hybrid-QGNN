READ ME

All code was ran on the UChicago Analysis Facility (see Jared's google doc) on the conda image. All files assume acorn environment is active, so follow install instructions on the acorn GitLab for that. I also use PennyLane for the quantum networks, which is external to acorn.

Note: all steps assume graph data is in directory named module_map, and is then further split into test/train/validate. Data I used is the same as example 1 of acorn, which is outdated. May want to obtain new data.


Ex: 'module_map/testset'

Training:

    Run qgnn_run.py to train, validate, and test a GNN. Currently quantum networks are implemented, but you can chose to run a classical net by setting qnn=False in "model = InteractionGNN(hparams, qnn = True).to(device)" from the qgnn_run.py file.

    qgnn_run.py will print out the model structure (which is taken directly from acorn), but feel free to look into 
    acorn_pennylane.py for details.

    Note on configuration: model setup requires use of a gnn_train.yaml, similar to Example 1 from acorn. However, only a
    few arguments are still required, namely: weighting, hard_cuts, and model_parameters. I use the yaml file 
    from Example 1, but many of its arguments are unused.

    TO RUN: python3 qgnn_run.py [yaml config file] [model name]
    EX: python3 qgnn_run.py gnn_train.yaml classic_gnn_5

    Where yaml config file is name of config, and model name is name used to identify training session. 
    Model name will be included in saved files.

    Check qgnn_terminal.txt to see an example ouput when running.


Inference:

    Run qgnn_infer.py to score test set with trained model. Scored graphs will be saved to scored_graphs_path which should be a folder.
    Model is loaded from file .pth or .pt file, which is passed into terminal.

    This uses a yaml file to cut graphs, but this can be the same as your training yaml. This implementation is mostly an artifact of acorn that I just didn't change

    TO RUN: python3 qgnn_infer.py [yaml config file] [model_path] [scored_graphs_path]
    EX: python3 qgnn_infer.py gnn_train.yaml epoch1_classic_gnn_5.pth classic_gnn_5_scores


Evaluation:

    Plotting is done in a Jupyter notebook. There are comments showing what to edit, but for completeness, 
    plot_config is a dictionary to edit in the notebook, and config is a gnn_eval.yaml file to edit.

    Notebook will save edgewise efficiency plots.

First steps in future work: I was never able to properly implement GPU compatibility with Pennylane and Pytorch working together, although it definitely should be possible. CPU vs GPU control is set in qgnn_run.py

Investigation of circuit depth and gate choices on model performance would be very interesting. Can the right choices surpass classical networks?

Feel free to contact me with questions if my messy code doesn't make sense. :)





