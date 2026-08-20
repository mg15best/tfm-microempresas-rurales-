import unittest

from copy import deepcopy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.models.freeze_ets_v2_predictions import ARTIFACT_COLUMNS, KEY_COLUMNS
from src.models.modeling_v2_common import load_modeling_v2_config
from src.visualization import dashboard_data
from src.visualization.dashboard_data import (
    BASELINE_ID,
    CANONICAL_ARTIFACT_PATH,
    CANONICAL_ARTIFACT_SHA256,
    CANONICAL_GENERATOR_COMMIT,
    CANONICAL_GITHUB_RUN_ID,
    CANONICAL_LOGICAL_SHA256,
    CANONICAL_METADATA_PATH,
    CANONICAL_METADATA_SHA256,
    CUTOFF_POLICY_ID,
    DashboardDataError,
    EVIDENCE_SCOPE,
    PREDICTION_COLUMN,
    SELECTED_MODEL_ID,
    CanonicalArtifactError,
    InvalidTerritoryError,
    PreparedCanonicalValidation,
    build_dashboard_context,
    calculate_operational_validation_metrics,
    calculate_territory_validation_metrics,
    get_territory_history,
    get_territory_validation_metrics,
    load_canonical_validation_bundle,
    load_gold_history,
    prepare_canonical_validation,
    validate_b5_lifecycle,
)


class TestCanonicalValidationBundle(unittest.TestCase):
    """Contrato fail-closed del artifact oficial B4C consumido por B5."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.bundle = load_canonical_validation_bundle(cls.config)

    def _copy_bundle(self, root: Path) -> tuple[Path, Path]:
        artifact = root / CANONICAL_ARTIFACT_PATH
        metadata = root / CANONICAL_METADATA_PATH
        artifact.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.bundle.artifact_path, artifact)
        shutil.copy2(self.bundle.metadata_path, metadata)
        return artifact, metadata

    def test_loader_opens_exact_official_bundle(self) -> None:
        bundle = self.bundle

        self.assertEqual(bundle.artifact_sha256, CANONICAL_ARTIFACT_SHA256)
        self.assertEqual(bundle.metadata_sha256, CANONICAL_METADATA_SHA256)
        self.assertEqual(
            bundle.logical_prediction_sha256,
            CANONICAL_LOGICAL_SHA256,
        )
        self.assertEqual(bundle.generator_commit_sha, CANONICAL_GENERATOR_COMMIT)
        self.assertEqual(bundle.github_run_id, CANONICAL_GITHUB_RUN_ID)
        self.assertTrue(
            bundle.metadata["artifact"]["official_canonical_artifact"]
        )
        self.assertEqual(
            bundle.metadata["github"]["github_sha"],
            bundle.generator_commit_sha,
        )

    def test_model_baseline_cutoff_and_panel_are_canonical(self) -> None:
        predictions = self.bundle.predictions

        self.assertEqual(self.bundle.selected_model_id, SELECTED_MODEL_ID)
        self.assertEqual(self.bundle.baseline_id, BASELINE_ID)
        self.assertEqual(self.bundle.cutoff_policy_id, CUTOFF_POLICY_ID)
        self.assertEqual(list(predictions.columns), ARTIFACT_COLUMNS)
        self.assertEqual(len(predictions), 1750)
        self.assertEqual(predictions["territory_id"].nunique(), 50)
        self.assertEqual(predictions["target_month_id"].nunique(), 35)
        self.assertFalse(predictions.duplicated(KEY_COLUMNS).any())

    def test_loader_does_not_require_git_directory(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self._copy_bundle(root)

            loaded = load_canonical_validation_bundle(
                self.config,
                project_root=root,
            )

        self.assertEqual(loaded.artifact_sha256, CANONICAL_ARTIFACT_SHA256)
        self.assertFalse((root / ".git").exists())

    def test_parquet_with_same_schema_but_different_bytes_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact, _ = self._copy_bundle(root)
            changed = self.bundle.predictions.copy()
            changed.loc[0, PREDICTION_COLUMN] += 1.0
            changed.to_parquet(
                artifact,
                engine="pyarrow",
                compression="zstd",
                index=False,
            )

            with self.assertRaisesRegex(CanonicalArtifactError, "SHA-256"):
                load_canonical_validation_bundle(
                    self.config,
                    project_root=root,
                )

    def test_corrupted_parquet_bytes_fail_closed(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact, _ = self._copy_bundle(root)
            with artifact.open("ab") as stream:
                stream.write(b"corruption")

            with self.assertRaisesRegex(CanonicalArtifactError, "SHA-256"):
                load_canonical_validation_bundle(
                    self.config,
                    project_root=root,
                )

    def test_altered_metadata_fails_closed(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            _, metadata = self._copy_bundle(root)
            text = metadata.read_text(encoding="utf-8")
            metadata.write_text(
                text.replace(
                    "official_canonical_artifact: true",
                    "official_canonical_artifact: false",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CanonicalArtifactError, "metadata"):
                load_canonical_validation_bundle(
                    self.config,
                    project_root=root,
                )

    def test_metadata_parquet_incompatibility_is_rejected(self) -> None:
        metadata = deepcopy(self.bundle.metadata)
        metadata["invariants"]["rows"] = 1749

        with self.assertRaisesRegex(
            CanonicalArtifactError,
            "invariantes",
        ):
            dashboard_data._validate_predictions(
                self.bundle.predictions,
                metadata,
            )

    def test_duplicate_keys_and_schema_drift_are_rejected(self) -> None:
        duplicated = self.bundle.predictions.copy()
        duplicated.loc[1, KEY_COLUMNS] = duplicated.loc[0, KEY_COLUMNS].to_numpy()
        with self.assertRaisesRegex(CanonicalArtifactError, "duplicadas"):
            dashboard_data._validate_predictions(
                duplicated,
                self.bundle.metadata,
            )

        drifted = self.bundle.predictions.drop(columns="clipping_applied")
        with self.assertRaisesRegex(CanonicalArtifactError, "schema"):
            dashboard_data._validate_predictions(
                drifted,
                self.bundle.metadata,
            )

    def test_config_cannot_repoint_the_official_hash(self) -> None:
        changed = deepcopy(self.config)
        changed["operational_selection"]["canonical_validation"][
            "artifact_sha256"
        ] = "0" * 64

        with self.assertRaisesRegex(CanonicalArtifactError, "Anclaje"):
            load_canonical_validation_bundle(changed)


class TestCanonicalMetricsAndDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.bundle = load_canonical_validation_bundle(cls.config)
        cls.gold = load_gold_history(cls.config)
        cls.prepared = prepare_canonical_validation(cls.bundle, cls.config)

    def test_lifecycle_distinguishes_history_and_operation(self) -> None:
        validate_b5_lifecycle(self.config)

        self.assertEqual(
            self.config["baseline"]["role"],
            "historical_validation_baseline",
        )
        self.assertEqual(
            self.config["operational_selection"]["status"],
            "provisional_validation_champion",
        )
        self.assertEqual(
            self.config["ets_candidate"]["screening_status"],
            "passed_screening",
        )
        self.assertFalse(
            self.config["operational_selection"][
                "independent_test_confirmed"
            ]
        )
        fallback = self.config["operational_selection"]["fallback"]
        self.assertEqual(fallback["model_id"], BASELINE_ID)
        self.assertEqual(fallback["policy"], "availability_only")
        self.assertFalse(fallback["performance_based"])

    def test_pooled_operational_metrics_match_canonical_evidence(self) -> None:
        metrics = calculate_operational_validation_metrics(
            self.bundle.predictions
        )

        self.assertEqual(metrics["n"], 1750)
        self.assertAlmostEqual(metrics["MAE"], 4084.574535196216)
        self.assertAlmostEqual(metrics["RMSE"], 7770.827125343509)
        self.assertAlmostEqual(metrics["WAPE_pct"], 19.68924738072576)
        self.assertAlmostEqual(metrics["bias"], -1862.1237643616084)

    def test_territory_metrics_use_operational_prediction(self) -> None:
        metrics = calculate_territory_validation_metrics(
            self.bundle.predictions,
            self.config,
        )

        self.assertEqual(len(metrics), 50)
        self.assertEqual(metrics["validation_rows"].unique().tolist(), [35])
        self.assertEqual(set(metrics["prediction_column"]), {PREDICTION_COLUMN})
        self.assertEqual(set(metrics["selected_model_id"]), {SELECTED_MODEL_ID})
        self.assertEqual(set(metrics["evidence_scope"]), {EVIDENCE_SCOPE})

        araba_rows = self.bundle.predictions.loc[
            self.bundle.predictions["territory_id"].eq("ES-PROV-01")
        ]
        expected_metrics = calculate_operational_validation_metrics(araba_rows)
        araba = metrics.loc[
            metrics["territory_id"].eq("ES-PROV-01")
        ].iloc[0]
        self.assertAlmostEqual(araba["validation_mae"], expected_metrics["MAE"])
        self.assertAlmostEqual(
            araba["validation_rmse"],
            expected_metrics["RMSE"],
        )
        self.assertAlmostEqual(
            araba["validation_bias"],
            expected_metrics["bias"],
        )
        self.assertEqual(araba["validation_rows"], expected_metrics["n"])
        expected = np.abs(
            araba_rows[PREDICTION_COLUMN] - araba_rows["actual"]
        ).sum() / np.abs(araba_rows["actual"]).sum() * 100
        actual = metrics.loc[
            metrics["territory_id"].eq("ES-PROV-01"),
            "validation_wape_pct",
        ].iloc[0]
        self.assertAlmostEqual(actual, expected)

    def test_prepared_metrics_cover_panel_and_public_boundary_is_closed(
        self,
    ) -> None:
        self.assertEqual(len(self.prepared.territory_metrics), 50)
        self.assertEqual(
            get_territory_validation_metrics(
                "ES-PROV-01",
                evidence=self.prepared,
                config=self.config,
            ).territory_name,
            "Araba/Álava",
        )
        with self.assertRaisesRegex(TypeError, "prepare_canonical_validation"):
            PreparedCanonicalValidation()
        with self.assertRaisesRegex(
            CanonicalArtifactError,
            "preparada",
        ):
            get_territory_validation_metrics(
                "ES-PROV-01",
                evidence=self.bundle,  # type: ignore[arg-type]
                config=self.config,
            )

    def test_zero_wape_denominator_is_explicitly_nan(self) -> None:
        metrics = calculate_operational_validation_metrics(
            pd.DataFrame(
                {
                    "actual": [0.0, 0.0],
                    PREDICTION_COLUMN: [0.0, 1.0],
                }
            )
        )

        self.assertTrue(np.isnan(metrics["WAPE_pct"]))
        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["MAE"], 0.5)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["bias"], 0.5)

    def test_badajoz_metrics_describe_selected_system_with_fallback(self) -> None:
        metrics = calculate_territory_validation_metrics(
            self.bundle.predictions,
            self.config,
        )
        badajoz = metrics.loc[
            metrics["territory_id"].eq("ES-PROV-06")
        ].iloc[0]

        self.assertEqual(badajoz["selected_model_id"], SELECTED_MODEL_ID)
        self.assertEqual(badajoz["candidate_available_rows"], 0)
        self.assertEqual(badajoz["availability_fallback_rows"], 35)
        self.assertEqual(badajoz["prediction_column"], PREDICTION_COLUMN)

    def test_dashboard_separates_operational_and_evaluation_lineages(self) -> None:
        context = build_dashboard_context(
            "ES-PROV-01",
            history_months=24,
            gold=self.gold,
            canonical_bundle=self.bundle,
            config=self.config,
        )
        lineage = context.lineage

        self.assertEqual(context.territory_name, "Araba/Álava")
        self.assertEqual(lineage.evaluation_scope, EVIDENCE_SCOPE)
        self.assertEqual(
            lineage.evaluation_artifact_sha256,
            CANONICAL_ARTIFACT_SHA256,
        )
        self.assertEqual(
            lineage.evaluation_logical_prediction_sha256,
            CANONICAL_LOGICAL_SHA256,
        )
        self.assertTrue(lineage.operational_source_snapshot_id)
        self.assertTrue(lineage.evaluation_source_snapshot_ids)
        self.assertFalse(hasattr(lineage, "evaluation_pipeline_run_id"))
        self.assertEqual(context.history["month_id"].iloc[0], "2024-07")
        self.assertEqual(context.history["month_id"].iloc[-1], "2026-06")

        newer = self.gold.copy()
        newer["source_snapshot_id"] = "newer-operational-snapshot"
        newer["pipeline_run_id"] = "newer-operational-run"
        newer["data_version"] = "newer-operational-version"
        newer_context = build_dashboard_context(
            "ES-PROV-01",
            gold=newer,
            canonical_bundle=self.bundle,
            config=self.config,
        )
        self.assertEqual(
            newer_context.lineage.operational_source_snapshot_id,
            "newer-operational-snapshot",
        )
        self.assertEqual(
            newer_context.lineage.evaluation_source_snapshot_ids,
            self.bundle.evaluation_source_snapshot_ids,
        )

    def test_invalid_territory_and_history_window_remain_blocking(self) -> None:
        with self.assertRaises(InvalidTerritoryError):
            build_dashboard_context(
                "ES-PROV-99",
                gold=self.gold,
                canonical_bundle=self.bundle,
                config=self.config,
            )
        with self.assertRaises(ValueError):
            get_territory_history(
                "ES-PROV-01",
                months=0,
                dataframe=self.gold,
                config=self.config,
            )

    def test_history_window_is_anchored_to_cutoff_and_preserves_gaps(self) -> None:
        gold = self.gold.loc[
            ~(
                self.gold["territory_id"].astype(str).eq("ES-PROV-01")
                & self.gold["month_id"].astype(str).eq("2022-10")
            )
        ].copy()

        history = get_territory_history(
            "ES-PROV-01",
            months=3,
            latest_available_month_id="2022-11",
            dataframe=gold,
            config=self.config,
        )

        self.assertEqual(history["month_id"].astype(str).tolist(), [
            "2022-09",
            "2022-11",
        ])

    def test_history_preserves_provisionality_from_gold(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            latest_available_month_id="2026-06",
            dataframe=self.gold,
            config=self.config,
        )
        source = self.gold.loc[
            self.gold["territory_id"].astype(str).eq("ES-PROV-01")
            & self.gold["month_id"].astype(str).eq("2026-06"),
            "is_provisional",
        ].iloc[0]

        self.assertEqual(
            history.loc[
                history["month_id"].astype(str).eq("2026-06"),
                "is_provisional",
            ].iloc[0],
            source,
        )

    def test_empty_gold_is_blocking(self) -> None:
        with self.assertRaises(DashboardDataError):
            get_territory_history(
                "ES-PROV-01",
                dataframe=self.gold.iloc[0:0],
                config=self.config,
            )

    def test_duplicate_gold_territory_month_is_blocking(self) -> None:
        duplicated = pd.concat(
            [self.gold, self.gold.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(DashboardDataError, "duplicadas"):
            get_territory_history(
                "ES-PROV-01",
                dataframe=duplicated,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
