import unittest

from dataclasses import replace
from datetime import date
from math import sqrt
from unittest.mock import patch

import pandas as pd

from src.application.forecast_service import (
    ProductCompositionError,
    build_forecast_product_context,
)
from src.models.inference import InvalidTerritoryError
from src.models.modeling_common import load_config
from src.visualization.dashboard_data import (
    load_official_validation_metrics,
    load_validation_predictions,
)


class TestForecastService(unittest.TestCase):
    """Pruebas unitarias de la composicion del producto."""

    def setUp(self) -> None:
        self.config = {
            "problem": {
                "territory_level": "province",
                "forecast_frequency": "monthly",
                "forecast_horizon_months": 1,
            },
            "target": {
                "source_column": "overnight_stays_total",
            },
            "source_dataset": {
                "path": "data/gold/gold_tourism_demand_monthly.parquet",
                "expected_territory_level": "province",
            },
            "modeling_dataset": {
                "path": "data/gold/gold_modeling_dataset_monthly.parquet",
                "forecast_horizon": 1,
            },
            "baseline": {
                "name": "seasonal_naive_lag_12",
                "prediction_feature": "lag_12_overnight_stays",
            },
            "fallback": {
                "if_no_candidate_beats_baseline": {
                    "selected_solution": "seasonal_naive_lag_12",
                },
            },
            "validation": {
                "folds": [{"name": "validation_1"}],
            },
        }
        self.gold = self._build_gold()
        self.predictions = self._build_predictions()
        self.official_metrics = pd.DataFrame(
            {
                "evaluation_split": ["validation_pooled"],
                "model": ["seasonal_naive_lag_12"],
                "rows": [2],
                "MAE": [15.0],
                "RMSE": [sqrt(250.0)],
                "WAPE_pct": [10.0],
                "mean_bias": [5.0],
            }
        )

    @staticmethod
    def _build_gold() -> pd.DataFrame:
        rows = [
            ("ES-PROV-01", "Provincia A", "2024-07", 80.0, False),
            ("ES-PROV-01", "Provincia A", "2025-09", 123.0, True),
            ("ES-PROV-01", "Provincia A", "2026-05", 500.0, True),
            ("ES-PROV-01", "Provincia A", "2026-06", 510.0, True),
            ("ES-PROV-02", "Provincia B", "2025-09", 456.0, True),
            ("ES-PROV-02", "Provincia B", "2026-06", 600.0, True),
        ]
        return pd.DataFrame(
            {
                "territory_id": [row[0] for row in rows],
                "territory_name": [row[1] for row in rows],
                "month_id": [row[2] for row in rows],
                "date_month": pd.to_datetime([row[2] for row in rows]),
                "overnight_stays_total": [row[3] for row in rows],
                "covid_period": False,
                "data_status": [
                    "provisional" if row[4]
                    else "final_or_not_marked_provisional"
                    for row in rows
                ],
                "is_provisional": [row[4] for row in rows],
                "coverage_quality": "high",
                "source_snapshot_id": "operational-snapshot",
                "pipeline_run_id": "operational-run",
                "data_version": "gold-v1",
            }
        )

    @staticmethod
    def _build_predictions() -> pd.DataFrame:
        observations = [
            ("ES-PROV-01", "Provincia A", 100.0, 90.0),
            ("ES-PROV-02", "Provincia B", 200.0, 220.0),
        ]
        records: list[dict[str, object]] = []
        for model in ("ridge", "hgb"):
            for territory_id, territory_name, actual, baseline in observations:
                records.append(
                    {
                        "territory_id": territory_id,
                        "territory_name": territory_name,
                        "target_month_id": "2023-01",
                        "target_date_month": pd.Timestamp("2023-01-01"),
                        "evaluation_split": "validation_1",
                        "source_snapshot_id": "evaluation-snapshot",
                        "pipeline_run_id": "evaluation-run",
                        "data_version": "modeling-v1",
                        "created_at": pd.Timestamp("2026-08-03", tz="UTC"),
                        "model": model,
                        "baseline_id": "seasonal_naive_lag_12",
                        "dataset_path": (
                            "data/gold/gold_modeling_dataset_monthly.parquet"
                        ),
                        "actual": actual,
                        "baseline_prediction": baseline,
                        "validation_start": "2023-01",
                        "structural_train_end": "2022-12",
                        "availability_train_end": "2022-10",
                        "effective_train_end": "2022-10",
                    }
                )
        return pd.DataFrame.from_records(records)

    def _build(self, territory_id: str = "ES-PROV-01", **kwargs: object):
        kwargs.setdefault("as_of_date", "2026-08-13")
        return build_forecast_product_context(
            territory_id,
            gold=self.gold,
            predictions=self.predictions,
            official_metrics=self.official_metrics,
            config=self.config,
            **kwargs,
        )

    def test_valid_territory_returns_composed_context(self) -> None:
        product = self._build()

        self.assertEqual(product.forecast.territory_id, "ES-PROV-01")
        self.assertEqual(product.dashboard.territory_id, "ES-PROV-01")
        self.assertEqual(product.forecast.territory_name, "Provincia A")
        self.assertEqual(product.dashboard.territory_name, "Provincia A")

    def test_invalid_territory_preserves_blocking_error(self) -> None:
        with self.assertRaises(InvalidTerritoryError):
            self._build("ES-PROV-99")

    def test_as_of_date_controls_forecast_origin_and_target(self) -> None:
        product = self._build()

        self.assertEqual(product.as_of_date, date(2026, 8, 13))
        self.assertEqual(product.forecast_origin_month_id, "2026-08")
        self.assertEqual(product.forecast.target_month_id, "2026-09")
        self.assertEqual(product.forecast.reference_month_id, "2025-09")
        self.assertEqual(product.forecast.latest_available_month_id, "2026-06")

    def test_gold_is_loaded_once_and_reused_by_both_layers(self) -> None:
        with patch(
            "src.application.forecast_service.load_gold_history",
            return_value=self.gold,
        ) as load_gold:
            product = build_forecast_product_context(
                "ES-PROV-01",
                as_of_date="2026-08-13",
                predictions=self.predictions,
                official_metrics=self.official_metrics,
                config=self.config,
            )

        load_gold.assert_called_once_with(self.config)
        self.assertEqual(product.forecast.source_snapshot_id, "operational-snapshot")
        self.assertEqual(
            product.dashboard.lineage.operational_source_snapshot_id,
            "operational-snapshot",
        )

    def test_warnings_are_preserved(self) -> None:
        product = self._build()

        self.assertTrue(product.forecast.reference_is_provisional)
        self.assertIn(
            "provisional_reference_data",
            {warning.code for warning in product.forecast.warnings},
        )

    def test_history_uses_a_natural_month_window(self) -> None:
        product = self._build(history_months=2)

        self.assertEqual(
            product.dashboard.history["month_id"].tolist(),
            ["2026-05", "2026-06"],
        )

    def test_mismatched_territory_is_blocking(self) -> None:
        valid = self._build()
        mismatched = replace(
            valid.dashboard,
            territory_id="ES-PROV-02",
            territory_name="Provincia B",
        )

        with patch(
            "src.application.forecast_service.build_dashboard_context",
            return_value=mismatched,
        ):
            with self.assertRaisesRegex(
                ProductCompositionError,
                "mismo territorio",
            ):
                self._build()

    def test_mismatched_operational_lineage_is_blocking(self) -> None:
        valid = self._build()
        mismatched_lineage = replace(
            valid.dashboard.lineage,
            operational_source_snapshot_id="another-snapshot",
        )
        mismatched = replace(valid.dashboard, lineage=mismatched_lineage)

        with patch(
            "src.application.forecast_service.build_dashboard_context",
            return_value=mismatched,
        ):
            with self.assertRaisesRegex(
                ProductCompositionError,
                "provenance operacional",
            ):
                self._build()


class TestForecastServiceIntegration(unittest.TestCase):
    """Integracion con la Gold operacional y evaluacion congelada."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.gold = pd.read_parquet(cls.config["source_dataset"]["path"])
        cls.predictions = load_validation_predictions()
        cls.official_metrics = load_official_validation_metrics()

    def _build(self, territory_id: str, *, history_months: int | None = None):
        return build_forecast_product_context(
            territory_id,
            as_of_date="2026-08-13",
            history_months=history_months,
            gold=self.gold,
            predictions=self.predictions,
            official_metrics=self.official_metrics,
            config=self.config,
        )

    def test_real_araba_context(self) -> None:
        product = self._build("ES-PROV-01", history_months=24)
        history_months = product.dashboard.history["date_month"].dt.to_period("M")

        self.assertEqual(product.forecast.territory_name, "Araba/Álava")
        self.assertEqual(product.forecast.target_month_id, "2026-09")
        self.assertEqual(product.forecast.reference_month_id, "2025-09")
        self.assertEqual(product.forecast.predicted_overnight_stays_total, 7_691.0)
        self.assertTrue(product.forecast.reference_is_provisional)
        self.assertEqual(product.forecast.latest_available_month_id, "2026-06")
        self.assertTrue(product.forecast.is_operational)
        self.assertAlmostEqual(
            product.dashboard.validation_metrics.validation_wape_pct,
            24.392342351526025,
        )
        self.assertEqual(product.dashboard.validation_metrics.validation_rows, 35)
        self.assertEqual(str(history_months.min()), "2024-07")
        self.assertEqual(str(history_months.max()), "2026-06")
        self.assertTrue(
            product.dashboard.lineage.operational_source_snapshot_id
            != product.dashboard.lineage.evaluation_source_snapshot_id
        )

    def test_real_context_is_available_for_all_fifty_provinces(self) -> None:
        territory_ids = sorted(self.gold["territory_id"].astype(str).unique())
        products = [self._build(territory_id) for territory_id in territory_ids]

        self.assertEqual(len(products), 50)
        self.assertEqual(
            {product.forecast.target_month_id for product in products},
            {"2026-09"},
        )
        self.assertEqual(
            {product.forecast.reference_month_id for product in products},
            {"2025-09"},
        )
        self.assertTrue(all(product.forecast.is_operational for product in products))
        self.assertTrue(all(not product.dashboard.history.empty for product in products))
        self.assertTrue(
            all(
                product.dashboard.validation_metrics.validation_rows > 0
                for product in products
            )
        )
        self.assertEqual(
            len({
                product.dashboard.lineage.operational_source_snapshot_id
                for product in products
            }),
            1,
        )
        self.assertEqual(
            len({
                product.dashboard.lineage.evaluation_source_snapshot_id
                for product in products
            }),
            1,
        )


if __name__ == "__main__":
    unittest.main()
