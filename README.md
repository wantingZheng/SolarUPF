SolarUPF is a unified multi-objective ensemble framework for distributed photovoltaic forecasting under limited-data conditions, designed to jointly improve deterministic point prediction and probabilistic interval estimation.

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
```bash
cd ZWT_PROJECT
bash scripts/run.sh
```