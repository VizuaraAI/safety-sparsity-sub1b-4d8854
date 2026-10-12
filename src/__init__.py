"""safety-sparsity-sub1b: does safety-token sparsity survive below 1B parameters?

Package layout (read in this order the first time):

    config.py       load and validate configs/default.yaml
    modeling.py     load a model from the ladder, format chats, generate
    data.py         build the harmful / benign / capability probe corpora
    refusal.py      decide whether a generated response is a refusal
    activations.py  hooks for MLP post-activations and residual-stream reads,
                    plus neuron ablation and directional ablation
    axis_token.py   Axis A, token-depth sparsity  -> k*
    axis_neuron.py  Axis B, neuron-support sparsity -> p*, and SCE
    axis_direction.py Axis C, subspace-rank sparsity -> r*
    capability.py   log-likelihood accuracy on ARC and HellaSwag
    metrics.py      the Safety Localization Index and the aggregation table
    evaluate.py     entry point: run all three axes for one model
    train.py        entry point: full SFT vs LoRA vs localized edit
"""

__version__ = "0.1.0"
