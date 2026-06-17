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

## Repository Structure

```
enose-pca-fingerprinting/
│
├── data/
│   └── raw/                        # Place UCI .dat files here (see below)
│
├── figures/                        # All output figures (300 DPI PNG)
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── scree_plot.png
│   ├── pca_2d_scatter.png
│   ├── drift_visualization.png
│   ├── drift_trajectories.png
│   └── loading_plot.png
│
├── 01_PCA_Drift_Analysis.ipynb     # Main analysis notebook
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

Download the UCI Gas Sensor Array Drift Dataset from:
https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip

Unzip and place the 10 batch files (`batch1.dat` to `batch10.dat`)
inside the `data/raw/` folder.

**Step 4 — Run the notebook**

Open `01_PCA_Drift_Analysis.ipynb` in Jupyter or Google Colab.
Run all cells from top to bottom.

Or open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adhlaphy/enose-pca-fingerprinting/blob/main/01_PCA_Drift_Analysis.ipynb)

---

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

### PCA — Mathematical Summary

Given normalized data matrix X of shape (N × p):

1. Center:         X̃ = X − mean(X)
2. Covariance:     C = (1/N−1) × X̃ᵀX̃
3. Eigendecompose: C × v = λv
4. Project:        Z = X̃ × V

Explained variance ratio: EVR = λₖ / Σλ

PCA was implemented manually using `numpy.linalg.eigh` and validated
against `sklearn.decomposition.PCA`. All 10 PC projections matched
to 8 decimal places.

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
capture 90% of variance — a 16× reduction from 128 dimensions.

### 4. PCA 2D Scatter
Ethylene and Acetone form tight, well-separated clusters.
Ethanol and Acetaldehyde show partial overlap, consistent with
their similar chemical structures and surface reaction mechanisms
on metal-oxide sensors.

### 5. Drift Visualization
Batch-colored projection reveals systematic displacement of all
gas clusters from early batches to late batches, confirming
temporal sensor drift over 36 months.

### 6. Drift Trajectories
Centroid trajectories show consistent directional movement in PC
space rather than random scatter. Acetone shows the largest total
drift. Toluene shows the least. Consistent direction indicates the
drift is systematic and potentially correctable.

### 7. Loading Plot
Sensor 8 contributes most strongly to PC1, appearing twice in the
top 5 loading rankings. This identifies Sensor 8 as the most
critical sensor for gas discrimination in this array.

---

## Limitations

- PCA is linear and may not capture nonlinear cluster structure
- Drift is characterized but not corrected
- Ethanol and Acetaldehyde overlap is unresolved at the PCA stage
- Feature extraction uses pre-computed UCI features, not raw transients

## Future Work

- Project 2: SVM and Random Forest classification on PCA-reduced data
- Project 3: Drift compensation via subspace alignment
- Project 4: UMAP vs PCA comparison using cluster quality metrics

---

## References

1. Persaud, K. & Dodd, G. (1982). Analysis of discrimination mechanisms
   in the mammalian olfactory system using a model nose.
   *Nature*, 299, 352–355.

2. Vergara, A. et al. (2012). Chemical gas sensor drift compensation
   using classifier ensembles.
   *Sensors and Actuators B*, 166–167, 320–329.

3. Röck, F. et al. (2008). Electronic nose: Current status and
   future trends. *Chemical Reviews*, 108(2), 705–725.
