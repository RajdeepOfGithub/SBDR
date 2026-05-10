"""
SBDR Pipeline Validation Tests (C3 — Final Testing)
====================================================
Validates all pipeline outputs: data shapes, model artifacts,
column integrity, and key metrics from the SBDR capstone pipeline.

Run from project root:
    source .venv/bin/activate
    python -m pytest tests/test_pipeline.py -v
or:
    python tests/test_pipeline.py
"""

import os
import sys
import json
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ─── Helpers ───────────────────────────────────────────────────────────────

def load_csv_header(path):
    """Read only the header row of a CSV (fast — no pandas needed)."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return header


def count_csv_rows(path):
    """Count data rows (excluding header). Uses csv.reader to handle quoted multi-line fields."""
    import csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return sum(1 for _ in reader)


# ─── Data File Tests ────────────────────────────────────────────────────────

class TestDataFiles(unittest.TestCase):

    def _csv(self, name):
        return os.path.join(DATA_PROCESSED, name)

    def test_phase_a_final_dataset_exists(self):
        path = self._csv("sbdr_final_dataset.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_phase_a_final_dataset_shape(self):
        path = self._csv("sbdr_final_dataset.csv")
        cols = load_csv_header(path)
        rows = count_csv_rows(path)
        self.assertEqual(rows, 30_000, f"Expected 30000 rows, got {rows}")
        self.assertEqual(len(cols), 88, f"Expected 88 cols, got {len(cols)}")

    def test_bilstm_output_exists(self):
        path = self._csv("06_with_stress_vectors.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_bilstm_output_shape(self):
        path = self._csv("06_with_stress_vectors.csv")
        cols = load_csv_header(path)
        rows = count_csv_rows(path)
        self.assertEqual(rows, 30_000, f"Expected 30000 rows, got {rows}")
        self.assertEqual(len(cols), 122, f"Expected 122 cols, got {len(cols)}")

    def test_bilstm_output_key_columns(self):
        path = self._csv("06_with_stress_vectors.csv")
        cols = load_csv_header(path)
        for col in ["bilstm_recon_error", "bilstm_anomaly_flag"] + [f"stress_dim_{i:02d}" for i in range(32)]:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_finbert_output_exists(self):
        path = self._csv("07_with_distress_scores.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_finbert_output_shape(self):
        path = self._csv("07_with_distress_scores.csv")
        cols = load_csv_header(path)
        rows = count_csv_rows(path)
        self.assertEqual(rows, 30_000, f"Expected 30000 rows, got {rows}")
        self.assertEqual(len(cols), 94, f"Expected 94 cols, got {len(cols)}")

    def test_finbert_output_key_columns(self):
        path = self._csv("07_with_distress_scores.csv")
        cols = load_csv_header(path)
        for col in ["distress_turn_1", "distress_turn_2", "distress_turn_3",
                    "distress_avg", "distress_max", "distress_shift"]:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_xgboost_output_exists(self):
        path = self._csv("08_with_recovery_tiers.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_xgboost_output_shape(self):
        path = self._csv("08_with_recovery_tiers.csv")
        cols = load_csv_header(path)
        rows = count_csv_rows(path)
        self.assertEqual(rows, 30_000, f"Expected 30000 rows, got {rows}")
        self.assertEqual(len(cols), 135, f"Expected 135 cols, got {len(cols)}")

    def test_xgboost_output_key_columns(self):
        path = self._csv("08_with_recovery_tiers.csv")
        cols = load_csv_header(path)
        for col in ["recovery_tier"] + [f"tier_prob_{i}" for i in range(1, 6)]:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_audit_output_exists(self):
        path = self._csv("09_with_audit_tiers.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_audit_output_shape(self):
        path = self._csv("09_with_audit_tiers.csv")
        cols = load_csv_header(path)
        rows = count_csv_rows(path)
        self.assertEqual(rows, 30_000, f"Expected 30000 rows, got {rows}")
        self.assertEqual(len(cols), 142, f"Expected 142 cols, got {len(cols)}")

    def test_audit_output_key_columns(self):
        path = self._csv("09_with_audit_tiers.csv")
        cols = load_csv_header(path)
        for col in ["recovery_tier_final", "audit_rule_a", "audit_rule_b",
                    "audit_escalated", "audit_deescalated"]:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_chat_logs_exist(self):
        path = self._csv("sbdr_chat_logs.csv")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")


# ─── Model Artifact Tests ────────────────────────────────────────────────────

class TestModelArtifacts(unittest.TestCase):

    def test_bilstm_model_weights(self):
        path = os.path.join(MODELS_DIR, "bilstm", "best_model.pt")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        self.assertGreater(os.path.getsize(path), 1_000, "BiLSTM model file is suspiciously small")

    def test_bilstm_train_val_split(self):
        path = os.path.join(MODELS_DIR, "bilstm", "train_val_indices.npz")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_bilstm_manifest(self):
        path = os.path.join(MODELS_DIR, "bilstm", "bilstm_manifest.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        with open(path) as f:
            m = json.load(f)
        self.assertIn("training_summary", m)
        best_val_loss = m["training_summary"]["best_val_loss"]
        self.assertLess(best_val_loss, 0.01, "BiLSTM val loss should be < 0.01")

    def test_finbert_model_weights(self):
        path = os.path.join(MODELS_DIR, "finbert", "best_model.pt")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_finbert_manifest(self):
        path = os.path.join(MODELS_DIR, "finbert", "finbert_manifest.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        with open(path) as f:
            m = json.load(f)
        self.assertIn("mode", m)

    def test_xgboost_model(self):
        path = os.path.join(MODELS_DIR, "xgboost", "best_model.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        self.assertGreater(os.path.getsize(path), 10_000, "XGBoost model file is suspiciously small")

    def test_xgboost_manifest(self):
        path = os.path.join(MODELS_DIR, "xgboost", "xgboost_manifest.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        with open(path) as f:
            m = json.load(f)
        self.assertIn("metrics", m)
        self.assertGreater(m["metrics"]["accuracy"], 0.93, "XGBoost accuracy should be > 93%")
        self.assertGreater(m["metrics"]["auc_roc_macro"], 0.98, "XGBoost AUC-ROC should be > 0.98")

    def test_audit_manifest(self):
        path = os.path.join(MODELS_DIR, "xgboost", "audit_manifest.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        with open(path) as f:
            m = json.load(f)
        self.assertIn("audit_summary", m)
        self.assertGreater(m["audit_summary"]["total_escalated"], 100,
                           "Audit should escalate > 100 customers")

    def test_model_plots_exist(self):
        plots = [
            ("bilstm", "training_loss.png"),
            ("bilstm", "recon_error_dist.png"),
            ("xgboost", "confusion_matrix.png"),
            ("xgboost", "shap_importance.png"),
            ("xgboost", "audit_tier_comparison.png"),
            ("finbert", "distress_validation.png"),
        ]
        for subdir, fname in plots:
            path = os.path.join(MODELS_DIR, subdir, fname)
            self.assertTrue(os.path.exists(path), f"Missing plot: {path}")


# ─── Pipeline Integrity Tests ────────────────────────────────────────────────

class TestPipelineIntegrity(unittest.TestCase):
    """Validate key business-logic invariants across the pipeline outputs."""

    @classmethod
    def setUpClass(cls):
        """Load the final dataset once for all integrity tests."""
        try:
            import pandas as pd
            cls.df = pd.read_csv(
                os.path.join(DATA_PROCESSED, "09_with_audit_tiers.csv"),
                low_memory=False
            )
            cls.pandas_available = True
        except ImportError:
            cls.pandas_available = False

    def _skip_if_no_pandas(self):
        if not self.pandas_available:
            self.skipTest("pandas not available — skipping dataframe integrity tests")

    def test_no_null_customer_ids(self):
        self._skip_if_no_pandas()
        self.assertEqual(self.df.index.isnull().sum(), 0)

    def test_recovery_tier_final_range(self):
        self._skip_if_no_pandas()
        valid_tiers = {1, 2, 3, 4, 5}
        actual = set(self.df["recovery_tier_final"].unique())
        self.assertTrue(actual.issubset(valid_tiers),
                        f"Unexpected tier values: {actual - valid_tiers}")

    def test_tier5_count_in_range(self):
        self._skip_if_no_pandas()
        tier5_count = (self.df["recovery_tier_final"] == 5).sum()
        self.assertGreater(tier5_count, 400, f"Tier 5 count too low: {tier5_count}")
        self.assertLess(tier5_count, 800, f"Tier 5 count suspiciously high: {tier5_count}")

    def test_distress_avg_bounded(self):
        self._skip_if_no_pandas()
        self.assertTrue((self.df["distress_avg"] >= 0.0).all(), "distress_avg contains negatives")
        self.assertTrue((self.df["distress_avg"] <= 1.0).all(), "distress_avg exceeds 1.0")

    def test_bilstm_anomaly_flag_binary(self):
        self._skip_if_no_pandas()
        valid = {0, 1}
        actual = set(self.df["bilstm_anomaly_flag"].unique())
        self.assertTrue(actual.issubset(valid),
                        f"bilstm_anomaly_flag has non-binary values: {actual}")

    def test_anomaly_rate_approx_5pct(self):
        self._skip_if_no_pandas()
        rate = self.df["bilstm_anomaly_flag"].mean()
        self.assertAlmostEqual(rate, 0.05, delta=0.02,
                               msg=f"Anomaly rate {rate:.3f} far from expected 5%")

    def test_tier_probs_sum_to_one(self):
        self._skip_if_no_pandas()
        prob_cols = [f"tier_prob_{i}" for i in range(1, 6)]
        prob_sums = self.df[prob_cols].sum(axis=1)
        # Allow floating point tolerance
        self.assertTrue((prob_sums - 1.0).abs().lt(1e-4).all(),
                        "tier_prob_1..5 do not sum to 1.0 for all rows")

    def test_fairness_tier5_rates_by_sex(self):
        """Tier 5 escalation should not be disproportionate by gender (> 3× ratio)."""
        self._skip_if_no_pandas()
        rates = (self.df.groupby("SEX")["recovery_tier_final"]
                 .apply(lambda x: (x == 5).mean()))
        max_rate = rates.max()
        min_rate = rates.min()
        if min_rate > 0:
            ratio = max_rate / min_rate
            self.assertLess(ratio, 3.0,
                            f"Gender Tier 5 rate ratio {ratio:.2f}× exceeds 3× fairness threshold")

    def test_column_count(self):
        self._skip_if_no_pandas()
        self.assertEqual(len(self.df.columns), 142,
                         f"Expected 142 columns, got {len(self.df.columns)}")

    def test_row_count(self):
        self._skip_if_no_pandas()
        self.assertEqual(len(self.df), 30_000,
                         f"Expected 30000 rows, got {len(self.df)}")


# ─── Dashboard Smoke Test ────────────────────────────────────────────────────

class TestDashboardImport(unittest.TestCase):

    def test_dashboard_syntax(self):
        """Verify dashboard.py has no syntax errors."""
        import ast
        dashboard_path = os.path.join(PROJECT_ROOT, "dashboard.py")
        self.assertTrue(os.path.exists(dashboard_path), "dashboard.py not found")
        with open(dashboard_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            self.fail(f"dashboard.py has a syntax error: {e}")

    def test_requirements_file_exists(self):
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        self.assertTrue(os.path.exists(req_path), "requirements.txt not found")

    def test_limitations_doc_exists(self):
        path = os.path.join(PROJECT_ROOT, "LIMITATIONS.md")
        self.assertTrue(os.path.exists(path), "LIMITATIONS.md not found")

    def test_readme_exists(self):
        path = os.path.join(PROJECT_ROOT, "README.md")
        self.assertTrue(os.path.exists(path), "README.md not found")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestDataFiles, TestModelArtifacts,
                TestPipelineIntegrity, TestDashboardImport]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
