# Decoding-the-Mirror-System-A-Computational-Journey-into-Brain-Connectivity-and-Emotional-Processing

## Project Description
This project focuses on the analysis of functional MRI (fMRI) data collected during an emotion recognition task. The primary objective is to study the role of the mirror system within the brain by mapping its key areas, analyzing BOLD signals, and comparing its organization with that of the entire brain. Additionally, graph-based models are developed to highlight the modular organization and connectivity of cerebral regions.

## Workflow Steps
__Data Import__

fMRI data were imported in NIfTI format, and the dataset dimensions were analyzed to ensure quality and consistency. This step provided an understanding of the 3D data structure (e.g., 256x256x256 voxels).

## Mapping the Mirror System
Using atlas-based brain masks, the regions of the mirror system were identified and mapped. This allowed a focused analysis on specific functional areas of interest.

## Visualization of Time Series
The BOLD (Blood Oxygen Level Dependent) signal was analyzed by comparing the time series of activations in the mirror system areas to those of other brain regions. This comparison provided insights into functional activations across contexts.

## Detailed Analysis of Mirror System Areas
Time series of the BOLD signal were extracted and analyzed separately for each region within the mirror system, identifying unique characteristics of each area.

## Correlation Matrix
A correlation matrix was constructed to compare the activity of the mirror system areas with all other regions defined in the brain atlas. This step helped identify significant functional relationships.

## Graph Construction

__Mirror System Graph__: A graph representing the connections within the mirror system areas.

__Global Graph__: A graph including all brain areas mapped in the atlas.
Graph Comparison

In the mirror system graph, a single connectivity module was identified using the Louvain clustering algorithm.
In the global graph, four distinct modules were identified:
Most areas of the mirror system belonged to the same module.
Three areas of the mirror system were part of a separate module, suggesting a unique functionality compared to the rest of the mirror system.

## Implications and Future Directions
__Pathology Prediction__

The findings of this study could support the development of predictive models for neuropsychiatric disorders such as autism and schizophrenia. Advanced machine learning techniques are proposed, including:

__Support Vector Machine (SVM)__,

__Random Forest__,

__Graph Neural Networks (GNN)__.

These approaches aim to predict pathological conditions based on the functional characteristics of the mirror system.
Computational Models of the Nervous System
Building computational models of the nervous system, informed by this study, is a crucial step toward understanding brain functions. These models could provide valuable tools for simulating and studying neural networks under both normal and pathological conditions.

## Technical Requirements
Python Libraries Used:
nibabel, numpy, pandas, matplotlib, nilearn, seaborn.
Brain Atlases:
Specific brain atlas used for defining and mapping regions of interest.
Usage Instructions
Run the notebook to import and preprocess the fMRI data.
Analyze BOLD signal time series with a focus on mirror system areas.
Construct and compare graphs to interpret modular results.
Use the findings to explore future applications, including neural network modeling and pathology prediction.

This README provides a comprehensive and discursive guide to the project, emphasizing its potential impact on neuroscience research and clinical applications.
