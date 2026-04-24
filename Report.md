# Self-Pruning Neural Network Report

## Overview

In this project, we implemented a **self-pruning neural network** where the model learns to remove its own unnecessary connections during training. Instead of performing pruning as a post-processing step, the network dynamically identifies and suppresses weak weights using a learnable gating mechanism.

---

## Prunable Linear Layer

Each weight in the network is associated with a learnable **gate parameter**. These gate scores are passed through a sigmoid function to constrain them between 0 and 1:

- If a gate value is close to **1**, the weight is active  
- If a gate value is close to **0**, the weight is effectively pruned  

The forward operation becomes:

$$
W_{pruned} = W \odot \sigma(G)
$$

where:

- $W$ = weight matrix  
- $G$ = gate scores  
- $\sigma(G)$ = sigmoid applied to gate scores  

This ensures:
- Gates are differentiable  
- Gradients flow through both weights and gate parameters  
- The model learns which connections are important  

---

## Sparsity Regularization

To encourage pruning, we introduce a sparsity penalty:

$$
Loss = ClassificationLoss + \lambda \cdot SparsityLoss
$$

- **ClassificationLoss**: Cross-Entropy Loss  
- **SparsityLoss**: Mean of all gate values  

### Why L1 encourages sparsity

The L1 norm penalizes non-zero values, pushing many gate values toward zero. Since gates lie between 0 and 1, minimizing them naturally suppresses weak connections, resulting in a sparse network.

---

## Experimental Setup

- **Dataset**: CIFAR-10  
- **Model**: CNN with prunable fully connected layers  
  - 2 Convolutional layers  
  - 2 PrunableLinear layers  
- **Activation**: ReLU  
- **Optimizer**: Adam  
- **Learning Rate**: 8e-4  
- **Epochs**: 20  
- **Batch Size**: 128  

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.05   | 68.52       | 61.53        |
| 0.1    | **69.44**   | **68.02**    |
| 0.2    | 67.81       | 67.08        |

---

## Analysis

From the results, we observe a clear trade-off between sparsity and accuracy:

- Increasing **λ** increases sparsity (more weights are pruned)  
- Very high λ may slightly reduce accuracy due to over-pruning  

### Key Observation

At **λ = 0.1**, the model achieves:
- Highest accuracy (**69.44%**)  
- High sparsity (**68.02%**)  

This indicates an optimal balance where:
- Redundant weights are removed  
- Important connections are preserved  

---

## Gate Distribution Visualization

![Gate Distribution](combined_gate_distribution.png)

### Interpretation

- Lower λ → gates spread out → less pruning  
- Higher λ → gates concentrate near **0** → more pruning  

A successful pruning model shows:
- A spike near **0** (pruned weights)  
- Some values away from 0 (important weights retained)  

---

## Limitations

- Only fully connected layers are pruned; convolutional layers remain dense  
- The CNN architecture is relatively simple  
- Sparsity tends to plateau beyond a certain λ  

---

## Conclusion

This project demonstrates that neural networks can learn to prune themselves during training using a gating mechanism and sparsity regularization.

The model successfully:
- Removes unnecessary connections dynamically  
- Achieves significant sparsity (~68%)  
- Maintains high classification accuracy (~69%)  

---

## Final Outcome

> The best-performing model (**λ = 0.1**) achieves an optimal balance between sparsity and accuracy, making it efficient and compact.

---

