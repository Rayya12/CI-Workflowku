# Workflow-CI — MLflow Project + GitHub Actions

## Struktur Repository
```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                          ← GitHub Actions workflow
└── MLProject/
    ├── modelling.py                        ← Training script
    ├── conda.yaml                          ← Conda environment
    ├── MLProject                           ← MLflow Project config
    ├── requirements.txt
    └── credit_risk_dataset_preprocessed.csv
```

## Cara Kerja CI
Workflow otomatis berjalan ketika:
- Ada **push** ke branch `main`
- **Manual trigger** via tombol "Run workflow" di tab Actions GitHub

### Langkah yang dijalankan:
1. Checkout repo
2. Setup Python 3.10
3. Install dependencies
4. Jalankan `mlflow run` pada folder MLProject
5. Upload `mlruns/` sebagai GitHub Actions Artifact (tersimpan 30 hari)

## Menjalankan Secara Lokal
```bash
cd MLProject
mlflow run . --env-manager=local -P n_estimators=100 -P max_depth=10
mlflow ui   # buka http://127.0.0.1:5000
```
