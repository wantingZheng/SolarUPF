# SolarUPF

**Paper Title:** *Cross-Site Point and Interval Forecasting of Distributed Photovoltaic Power under Limited-Data Conditions*

SolarUPF is a unified multi-objective ensemble framework for distributed photovoltaic forecasting under limited-data conditions. It is designed to jointly improve **deterministic point prediction** and **probabilistic interval estimation**, with a particular focus on **cross-site generalization** under heterogeneous station conditions.

---

## Overview

Distributed photovoltaic (PV) forecasting often suffers from limited historical data, heterogeneous site conditions, and insufficient evaluation across geographically diverse stations. SolarUPF is developed to address these challenges through a unified framework for joint point and interval forecasting.

The framework is intended for:

- deterministic point forecasting of PV power output,
- probabilistic interval forecasting with uncertainty quantification,
- cross-site evaluation under limited-data conditions,
- feature augmentation with solar-geometry information.

---

## Framework

The overall framework of SolarUPF is illustrated below.

![SolarUPF Framework](assets/framework.png)

---

## Repository Structure

```text
SolarUPF/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ sample/
│  └─ processed/
├─ models/
├─ scripts/
├─ utils/
├─ evaluation/
├─ notebooks/
│  ├─ 01_compare_tabular_vs_timeseries.ipynb
│  ├─ 02_feature_augmentation.ipynb
│  ├─ 03_data_preprocessing_external_test.ipynb
│  └─ 04_shap_analysis.ipynb
├─ assets/
│  └─ framework.png
└─ results/
