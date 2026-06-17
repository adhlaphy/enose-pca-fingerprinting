# Electronic Nose Gas Fingerprinting via PCA

> PCA-based dimensionality reduction and sensor drift characterization
> of metal-oxide sensor array data for volatile organic compound discrimination.

**Author:** Adhla P  
**Affiliation:** BS-MS Physics, IISER Thiruvananthapuram (2025)  
**Dataset:** UCI Gas Sensor Array Drift Dataset  
**Tools:** Python 3.10 · NumPy · scikit-learn · Matplotlib · Seaborn  

---

## Overview

Electronic nose (e-nose) systems use arrays of cross-sensitive metal-oxide
sensors to identify gases through pattern recognition rather than absolute
selectivity. This project applies Principal Component Analysis (PCA) to the
UCI Gas Sensor Array Drift Dataset to:

1. Demonstrate gas-class discrimination in reduced PC space
2. Characterize sensor drift over 36 months across 10 temporal batches
3. Identify which sensors contribute most to chemical discrimination

---

## Key Results

| Finding | Value |
|---|---|
| Dataset size | 13,910 samples · 6 gases · 128 features · 10 batches |
| Variance explained by PC1 | 53.5% |
| Variance explained by PC1 + PC2 | 68.6% |
| PCs needed for 90% variance | 8 (16x dimensionality reduction) |
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
classifier performance and is an open research problem.

---

## Repository Structure

```
enose-pca-fingerprinting/
│
├── data/
│   └── raw/                      # Place UCI .dat files here (see below)
│
├── figures/
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── scree_plot.png
│   ├── pca_2d_scatter.png
│   ├── drift_visualization.png
│   ├── drift_trajectories.png
│   └── loading_plot.png
│
├── 01_PCA_Drift_Analysis.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation and Usage

**Step 1 — Clone the repository**
```bash
git clone https://github.com/adhlaphy/enose-pca-fingerprinting
cd enose-pca-fingerprinting
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset**

Download from:
https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip

Unzip and place batch1.dat through batch10.dat inside the data/raw/ folder.

**Step 4 — Run the notebook**

Open 01_PCA_Drift_Analysis.ipynb in Jupyter or Google Colab.
Run all cells from top to bottom.

---
## Mathematical Background

PCA is performed on the normalized data matrix **X** of shape (N x p)
where N = 13,910 samples and p = 128 features.

**Step 1 — Center the data:**
Each feature column is shifted to have zero mean.
X_centered = X - mean(X, axis=0)

**Step 2 — Covariance matrix:**
C = (1 / (N-1)) * X_centered.T @ X_centered
Shape: (128 x 128). Entry C[i,j] measures how features i and j vary together.

**Step 3 — Eigendecomposition:**
C @ v = lambda * v
Eigenvectors v are the principal component directions.
Eigenvalues lambda measure variance along each direction.

**Step 4 — Sort and project:**
Eigenvectors are sorted by descending eigenvalue.
Z = X_centered @ V_k  where V_k contains the top k eigenvectors.

**Explained variance ratio:**
EVR_k = lambda_k / sum(lambda)

**Validation:**
Manual implementation using numpy.linalg.eigh was compared against
sklearn.decomposition.PCA. All 10 principal component projections
matched to 8 decimal places, and explained variance ratios agreed
to 4 decimal places across all components.
## Methodology

```
Raw .dat files (10 batches, 36 months)
        ↓
Parse and stack → X: (13910, 128), y: (13910,)
        ↓
Z-score normalization (mean=0, std=1 per feature)
        ↓
PCA from scratch in NumPy — validated against sklearn to 8 decimal places
        ↓
Scree plot → 8 PCs needed for 90% variance
        ↓
2D scatter plot → gas clusters with 95% confidence ellipses
        ↓
Batch coloring → drift revealed across 10 time periods
        ↓
Centroid trajectories → drift quantified per gas over 36 months
        ↓
Loading analysis → Sensor 8 identified as primary discriminating sensor
```

---

## Figures

### 1. Feature Distributions
Single sensor features show heavily overlapping distributions across
all 6 gas classes, demonstrating that individual sensors cannot
discriminate gases and motivating the array-based e-nose approach.

### 2. Correlation Heatmap
High inter-sensor correlation confirms significant redundancy in the
128-dimensional feature space, justifying dimensionality reduction via PCA.

### 3. Scree Plot
PC1 explains 53.5% of total variance. Eight principal components
capture 90% of variance — a 16x reduction from 128 dimensions.

### 4. PCA 2D Scatter
Ethylene and Acetone form tight, well-separated clusters.
Ethanol and Acetaldehyde show partial overlap, consistent with
their similar chemical structures and surface reaction mechanisms.

### 5. Drift Visualization
Batch-colored projection reveals systematic displacement of all
gas clusters from early to late batches, confirming temporal drift
over 36 months.

### 6. Drift Trajectories
Centroid trajectories show consistent directional movement in PC
space. Acetone shows the largest total drift. Toluene shows the least.

### 7. Loading Plot
Sensor 8 contributes most strongly to PC1, identifying it as the
most critical sensor for gas discrimination in this array.

---

## Limitations

- PCA is linear and may not capture nonlinear cluster structure
- Drift is characterized but not corrected
- Ethanol and Acetaldehyde overlap is unresolved at the PCA stage

## Future Work

- Project 2: SVM and Random Forest classification
- Project 3: Drift compensation via subspace alignment
- Project 4: UMAP vs PCA comparison

---

## References

1. Persaud, K. & Dodd, G. (1982). Nature, 299, 352-355.
2. Vergara, A. et al. (2012). Sensors and Actuators B, 166-167, 320-329.
3. Röck, F. et al. (2008). Chemical Reviews, 108(2), 705-725.
