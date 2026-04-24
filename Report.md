Self-Pruning Neural Network Report
Overview

In this project, we implemented a self-pruning neural network where the model learns to remove its own unnecessary connections during training. Instead of performing pruning as a post-processing step, the network dynamically identifies and suppresses weak weights using a learnable gating mechanism.

Prunable Linear Layer

Each weight in the network is associated with a learnable gate parameter. These gate scores are passed through a sigmoid function to constrain them between 0 and 1:

If a gate value is close to 1, the weight is active
If a gate value is close to 0, the weight is effectively pruned

The forward operation becomes:

W
pruned
	​

=W⊙σ(G)

where:

W = weight matrix
G = gate scores
σ(G) = sigmoid applied to gate scores

This design ensures that:

Gates are differentiable
Gradients flow through both weights and gate parameters
The network can learn which connections to keep or remove
Sparsity Regularization

To encourage pruning, we introduce an L1-based penalty on the gate values:

Loss=ClassificationLoss+λ⋅SparsityLoss
ClassificationLoss: Cross-Entropy Loss
SparsityLoss: Mean of all gate values
Why L1 encourages sparsity

The L1 norm penalizes non-zero values, pushing many gate values toward zero. Since gates lie between 0 and 1, minimizing them naturally suppresses weak connections, leading to a sparse network.

Experimental Setup
Dataset: CIFAR-10
Model: CNN with prunable fully connected layers
2 Convolutional layers (feature extraction)
2 PrunableLinear layers (self-pruning)
Activation: ReLU
Optimizer: Adam
Learning Rate: 8e-4
Epochs: 20
Batch Size: 128
Results
Lambda	Accuracy (%)	Sparsity (%)
0.05	68.52	61.53
0.1	69.44	68.02
0.2	67.81	67.08
Analysis

From the results, we observe a clear trade-off between sparsity and accuracy:

As λ increases, sparsity increases, meaning more weights are pruned
However, very high λ can slightly reduce accuracy due to over-pruning
Key Observation
At λ = 0.1, the model achieves:
Highest accuracy (69.44%)
High sparsity (68.02%)

This indicates an optimal balance, where:

Redundant weights are removed
Important connections are preserved
Gate Distribution Visualization

The gate distribution shows:

Minimum ≈ 0.0009 → many weights effectively pruned
Mean ≈ 0.07–0.08 → overall sparsity pressure
Maximum ≈ 0.90+ → important connections retained
Interpretation
Lower λ → gates more spread → less pruning
Higher λ → gates concentrate near 0 → more pruning
A successful pruning model shows:
A spike near 0 (pruned weights)
Some values away from 0 (important weights retained)
Limitations
Only fully connected layers are pruned; convolutional layers remain dense
CNN architecture is relatively simple, limiting maximum achievable accuracy
Sparsity plateau is observed beyond a certain λ value
Conclusion

This project demonstrates that neural networks can learn to prune themselves during training using a gating mechanism and sparsity regularization.

The model successfully:

Learns to remove unnecessary weights dynamically
Achieves significant sparsity (~68%)
Maintains high classification accuracy (~69%)