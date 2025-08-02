# Temporal and Spectral Feature Extraction from EEG-Based Reaction Time using a Custom State-Space Model (MATLAB)

This repository contains MATLAB code implementing a custom State-Space Model (SSM) for extracting meaningful temporal and spectral features from reaction time (RT) data recorded during EEG-based behavioral experiments.

The code allows researchers to denoise trial-by-trial RTs, infer underlying latent dynamics using a linear SSM, and extract power spectral signatures from smoothed trajectories using multi-taper spectral analysis.

---

## Project Objective

Reaction time data from cognitive tasks is inherently noisy due to cognitive fluctuations, motor execution variability, and individual processing differences. These noise sources can obscure experimental effects (e.g., block changes or conflict trials).

This model enables:
- Temporal smoothing of RTs using a principled dynamical system
- Inference of latent behavioral states across time
- Spectral decomposition of RT dynamics to reveal rhythmic patterns (e.g., at ~9 Hz)

---

## Key Features

- Linear state-space model fit via Expectation-Maximization (EM)
- Kalman filter & smoother for forward/backward inference
- Trial-by-trial posterior estimation of smoothed RT
- Multi-taper Power Spectral Density (PSD) estimation on latent states
- Clear group-level rhythmicity differences in extracted features

---

## Example Results

Fig. 1 – Smoothed RT Dynamics  
- Posterior mean of latent states shows clear rhythmic modulation in low-Sucidal index participants  
- High-Sucidal index participants exhibit flatter or disrupted trajectories
- <img width="1250" height="625" alt="Low_SI_High_SI_RT" src="https://github.com/user-attachments/assets/64ee31d3-9faf-472a-b049-c87fa10edd0b" />


📈 Fig. 2 – Power Spectral Density (PSD)  
- Multi-taper PSD of smoothed RT shows a distinct ~9 Hz peak in low-SI participants  
- ~2.3 dB higher power around 9 Hz (1.7× increase) relative to high-SI participants
<img width="1059" height="626" alt="PSD_Old" src="https://github.com/user-attachments/assets/9d2603c2-ae4b-4d65-acfa-21c536b9b825" />

---

## 📁 Repository Structure
project-root/
├── COMPASS-master/ % Toolbox dependency (SSM modeling)
├── COMPASS_StateSpaceToolbox/ % Toolbox dependency (SSM modeling)
├── Final_MATLAB/
│ ├── n97_dataset.mat % EEG-derived RT data for 97 participants
│ ├── SSM_Fit.m % Main script: fits SSM model to RT data
├── README.md

