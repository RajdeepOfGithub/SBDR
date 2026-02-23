Phase B deliverables :

06 with_stress_vectors.csv — original data + 34 new columns
bilstm_manifest.json — documents your model config & column names
models/bilstm/best_model.pt — trained model checkpoint
Training plots showing convergence

can now use stress_dim_00 through stress_dim_31 columns (plus bilstm_recon_error and bilstm_anomaly_flag) as features in XGBoost.
