# Electronic Nose Gas Fingerprinting via PCA

> PCA-based dimensionality reduction and sensor drift characterization
> of metal-oxide sensor array data for volatile organic compound discrimination.

**Author:** Adhil P — BS-MS Physics, IISER Thiruvananthapuram (2025)
**Dataset:** UCI Gas Sensor Array Drift Dataset
**Tools:** Python · NumPy · scikit-learn · Matplotlib · Seaborn

---

## Overview

Electronic nose (e-nose) systems use arrays of cross-sensitive metal-oxide
sensors to identify gases through pattern recognition rather than absolute
selectivity. This project applies Principal Component Analysis (PCA) to the
UCI Gas Sensor Array Drift Dataset to:

1. Demonstrate gas-class discrimination in reduced PC space
2. Characterize sensor drift over 36 months across 10 temporal batches
3. Identify which sensors contribute most to chemical discrimination

This project was built as part of a self-directed machine learning for
gas sensing curriculum in preparation for PhD applications in sensor
informatics and computational materials science.

---

## Key Results

| Finding | Value |
|---|---|
| Dataset size | 13,910 samples · 6 gases · 128 features · 10 batches |
| Variance explained by PC1 | 53.5% |
| Variance explained by PC1 + PC2 | 68.6% |
| PCs needed for 90% variance | 8 (16× dimensionality reduction) |
| Gas with largest drift | Acetone |
| Gas with least drift | Toluene |
| Most influential sensor (PC1) | Sensor 8 |

---

## Scientific Background

### The Cross-Sensitivity Problem
A single metal-oxide sensor responds to multiple gases — this is the
cross-sensitivity problem. The solution is to use an array of sensors
with different but overlapping selectivities, creating a response pattern
unique to each gas. This is the electronic nose principle, first
demonstrated by Persaud and Dodd (1982).

### Why PCA?
With 16 sensors and 8 features per sensor, each gas exposure produces a
128-dimensional feature vector. PCA finds the directions of maximum
variance in this high-dimensional space and projects the data onto a
small number of orthogonal axes, enabling visualization and discrimination.

### Sensor Drift
Metal-oxide sensor responses change over time due to surface contamination,
grain growth, and baseline shifts. This causes the PC-space cluster of a
given gas to shift position over months. Uncorrected drift degrades
classifier performance over time and is an open research problem.

---

## Methodology
### PCA — Mathematical Summary

Given normalized data matrix X of shape (N × p):

1. Center:         X̃ = X − mean(X)
2. Covariance:     C = (1/N−1) × X̃ᵀX̃
3. Eigendecompose: C × v = λv
4. Project:        Z = X̃ × V

Explained variance ratio: EVR = λₖ / sum(λ)

PCA was first implemented manually using numpy.linalg.eigh and
then validated against sklearn.decomposition.PCA.
All 10 principal component projections matched to 8 decimal places.

---

## Figures

### 1. Feature Distributions
Single sensor features show heavily overlapping distributions across
all 6 gas classes. This demonstrates that individual sensors cannot
discriminate gases and motivates the array-based approach.

### 2. Correlation Heatmap
High inter-sensor correlation confirms significant redundancy in the
128-dimensional feature space, justifying dimensionality reduction.

### 3. Scree Plot
PC1 explains 53.5% of total variance. Eight principal components
capture 90% of variance — a 16× reduction from 128 dimensions.

### 4. PCA 2D Scatter
Ethylene and Acetone form tight, well-separated clusters.
Ethanol and Acetaldehyde show partial overlap, consistent with
their similar chemical structures and reaction mechanisms on
metal-oxide surfaces.

### 5. Drift Visualization
Batch-colored projection reveals systematic displacement of all
gas clusters from early batches to late batches, confirming
temporal sensor drift over 36 months.

### 6. Drift Trajectories
Centroid trajectories show consistent directional movement in
PC space. Acetone shows the largest total drift. Toluene shows
the least. Consistent direction indicates drift is systematic
and potentially correctable.

### 7. Loading Plot
Sensor 8 contributes most strongly to PC1, appearing twice in
the top 5 loading rankings. This identifies Sensor 8 as the
most critical sensor for gas discrimination.

---

## Repository Structure
---

## Installation

```bash
git clone https://github.com/adhlaphy/enose-pca-fingerprinting
cd enose-pca-fingerprinting
pip install -r requirements.txt
```

Then open `notebooks/01_PCA_Drift_Analysis.ipynb` in Jupyter or Colab.

---

## Limitations

- PCA is linear and may not capture nonlinear cluster structure
- Drift is characterized but not corrected
- Ethanol and Acetaldehyde overlap is unresolved at the PCA stage

## Future Work

- Project 2: SVM and Random Forest classification
- Project 3: Drift compensation via subspace alignment
- Project 4: UMAP comparison with PCA

---

## References

1. Persaud, K. & Dodd, G. (1982). Nature, 299, 352–355.
2. Vergara, A. et al. (2012). Sensors and Actuators B, 166–167, 320–329.
3. Röck, F. et al. (2008). Chemical Reviews, 108(2), 705–725.
