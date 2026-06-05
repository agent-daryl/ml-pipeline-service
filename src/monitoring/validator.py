from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, baseline_stats: dict[str, Any]):
        self.feature_means = baseline_stats["feature_means"]
        self.feature_stds = baseline_stats["feature_stds"]
        self.feature_mins = baseline_stats["feature_mins"]
        self.feature_maxs = baseline_stats["feature_maxs"]

    def validate_single(self, features: dict[str, float]) -> dict[str, Any]:
        warnings = []
        feature_checks = {}
        drift_contributors = []

        for name, value in features.items():
            if name not in self.feature_means:
                warnings.append(f"Unknown feature: {name}")
                feature_checks[name] = {"in_range": False, "z_score": 0.0}
                continue

            fmin = self.feature_mins[name]
            fmax = self.feature_maxs[name]
            mean = self.feature_means[name]
            std = self.feature_stds[name]

            in_range = fmin <= value <= fmax
            z_score = abs((value - mean) / std) if std > 0 else 0.0

            if not in_range:
                warnings.append(f"{name}={value} outside training range [{fmin:.2f}, {fmax:.2f}]")
            elif z_score > 3.0:
                warnings.append(f"{name}={value} is a statistical outlier (z={z_score:.2f})")

            feature_checks[name] = {"in_range": in_range, "z_score": round(z_score, 4)}
            drift_contributors.append(z_score)

        drift_score = sum(drift_contributors) / len(drift_contributors) if drift_contributors else 0.0
        is_valid = len(warnings) == 0 and drift_score < 2.0

        return {
            "is_valid": is_valid,
            "drift_score": drift_score,
            "warnings": warnings,
            "feature_checks": feature_checks,
        }

    def validate_batch(self, features_list: list[dict[str, float]]) -> dict[str, Any]:
        reports = [self.validate_single(f) for f in features_list]
        overall_valid = all(r["is_valid"] for r in reports)
        avg_drift = sum(r["drift_score"] for r in reports) / len(reports) if reports else 0.0
        all_warnings = []
        for r in reports:
            all_warnings.extend(r["warnings"])

        return {
            "is_valid": overall_valid,
            "avg_drift_score": round(avg_drift, 4),
            "total_warnings": len(all_warnings),
            "warnings": all_warnings[:20],
            "samples_valid": sum(1 for r in reports if r["is_valid"]),
            "samples_total": len(reports),
        }
