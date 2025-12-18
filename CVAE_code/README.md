# Clustering Atomic-Resolution STEM Images using CVAE-Based Anomaly Detection
This repository contains all the files necessary to reproduce the unsupervised clustering of atomic-resolution (S)TEM images based on defect types, using a CVAE-based anomaly detection and feature extraction pipeline, followed by a three-tier feature selection and k-means clustering workflow.

# 🔹 1. Clustering CdTe Dataset
# STEP 1: Training the CVAE Model for CdTe
To train the CVAE model and generate predicted images:

Download:

CdTe_CVAE_Model_Training.py

CdTe_Training_Image.tif (the training image)

Update the following paths inside the Python script:

Path to the working directory

Path to the training image (CdTe_Training_Image.tif)

Run the script.
The trained CVAE model will be saved in your working directory. This model will later be used to reconstruct bulk-like predicted images during the clustering process.

# STEP 2: Clustering the STEM Dataset
Download:

The folder DataSet_CdTe from the Image Data directory (contains 1119 .tif images)

The Jupyter notebook Clustering_CdTe_Data.ipynb

Update the paths in the notebook:

Path to the input image

Path to the CVAE model saved in Step 1

Path to the image folder DataSet_CdTe

The notebook includes the following stages:
Feature Extraction
Generate difference and filtered difference images using the CVAE model, and extract 47 pixel-intensity-based, shape-based, and frequency-based features.

Three-Tier Feature Selection

Pearson Correlation Filtering: Remove highly correlated features (threshold > 0.95)

Silhouette Score Filtering: Discard features with negative or low discriminative scores

Variance Thresholding: Remove features with variance < 0.1

Dimensionality Reduction
Apply Principal Component Analysis (PCA) to the shortlisted features.

Clustering
Perform k-means clustering, optimizing the number of clusters using silhouette score maximization.

Visualization
Use t-distributed Stochastic Neighbor Embedding (t-SNE) for 2D visualization of clustering results.

The final feature dataset can also be exported as a .csv file for future use.

# 🔹 2. Clustering SrTiO₃ (STO) Dataset
Repeat the same process as above, substituting CdTe with STO.

STEP 1: CVAE Model Training for STO
Download:

STO_CVAE_Model_Training.py

STO_Training_Image.jpg

Update paths inside the script as needed and run it to generate the trained CVAE model.

STEP 2: Clustering the STO Dataset
Download:

The folder DataSet_STO (contains 764 .tif images)

The notebook Clustering_STO_Data.ipynb

Update paths to:

The STO input image

The trained STO CVAE model

The dataset folder DataSet_STO

Follow the same pipeline for feature extraction, filtering, PCA, clustering, and visualization.

# Requirements
All code was implemented using Python (v3.12.3). The Convolutional Variational Autoencoder (CVAE) model was developed using TensorFlow (v2.18.0) and Keras (v3.7.0). Core scientific computing libraries include:

NumPy (v2.3.0)

SciPy (v1.15.3)

scikit-learn (v1.5.1)

Image processing and analysis were conducted using:

OpenCV (v4.10.0.84)

scikit-image (v0.25.0)

Pillow (v10.4.0)

All visualizations were generated using Matplotlib (v3.9.2).
HDF5-based model saving/loading was handled via h5py (v3.11.0).
Interactive development and execution were carried out using Jupyter Notebook (v6.5.7).
