from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tomllib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_platform.backtest.engine.hft_engine import (
    HftBacktestConfig,
    HftBacktestEngine,
    HftEngineConfig,
)
from trading_platform.backtest.io.s3_adapter import OCIObjectStorageS3Shim
from trading_platform.backtest.runtime.context import SweepContext
from trading_platform.core.risk.risk_config import RiskConfig
from trading_platform.strategies.strategy_config import StrategyConfig


class SweepMaterializer:
    """
    Materializes sweep input data from S3 into a local scratch directory.
    """

    def __init__(
        self,
        *,
        bucket: str,
    ) -> None:
        self._s3 = OCIObjectStorageS3Shim(region="eu-frankfurt-1")
        self._bucket = bucket

    def materialize(self, ctx: SweepContext) -> None:
        """
        Ensure all input files for the sweep are present locally.

        This operation is idempotent.
        """
        data_dir = ctx.scratch_data_dir
        ready_marker = data_dir / "_READY"

        if ready_marker.exists():
            return

        data_dir.mkdir(parents=True, exist_ok=True)

        for key in ctx.file_keys:
            filename = Path(key).name
            target_path = data_dir / filename

            if target_path.exists():
                continue

            self._s3.download_to_file(
                bucket=self._bucket,
                key=key,
                destination=target_path,
            )

        ready_marker.touch()


class SweepEngineRunner:
    """
    Runs exactly one HFT backtest sweep.

    One runner instance == one sweep == one engine.run().
    """

    def __init__(
        self,
        *,
        engine_cfg: HftEngineConfig,
        strategy_cfg: StrategyConfig,
        risk_cfg: RiskConfig,
    ) -> None:
        self._engine_cfg = engine_cfg
        self._strategy_cfg = strategy_cfg
        self._risk_cfg = risk_cfg

    def run(self, ctx: SweepContext) -> dict[str, Any]:
        """
        Execute the backtest for this sweep.

        Returns lightweight metadata about the run.
        """
        results_dir = ctx.scratch_results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT:
        # Engine expects a FIXED list of local file paths.
        data_files = [
            str(ctx.scratch_data_dir / Path(key).name)
            for key in ctx.file_keys
        ]

        engine_cfg = self._build_engine_cfg(data_files, results_dir)

        # Defensive: numpy will not create parent directories for output files.
        Path(engine_cfg.stats_npz_path).parent.mkdir(parents=True, exist_ok=True)

        backtest_cfg = HftBacktestConfig(
            # Keep IDs filesystem-safe. Some engines/libraries may use the ID
            # as part of output paths.
            id=f"{ctx.experiment_id}__{ctx.segment_id}__{ctx.sweep_id}",
            description="sweep execution",
            engine_cfg=engine_cfg,
            strategy_cfg=self._strategy_cfg,
            risk_cfg=self._risk_cfg,
        )

        engine = HftBacktestEngine(backtest_cfg)

        # Ensure any relative writes performed by the engine end up inside the
        # scratch subtree of this sweep.
        previous_cwd = Path.cwd()
        try:
            os.chdir(ctx.scratch_segment_dir)
            result = engine.run()
        finally:
            os.chdir(previous_cwd)

        done_marker = ctx.scratch_results_dir / "_DONE"
        done_marker.touch()

        return {
            "experiment_id": ctx.experiment_id,
            "segment_id": ctx.segment_id,
            "sweep_id": ctx.sweep_id,
            "stats_file": result.stats_file,
            "extra_metadata": result.extra_metadata,
        }

    def _build_engine_cfg(
        self,
        data_files: list[str],
        results_dir: Path,
    ) -> HftEngineConfig:
        """
        Clone the base engine config and inject sweep-specific paths.
        """
        cfg = replace(self._engine_cfg)

        # THIS is the critical binding to the engine semantics
        cfg.data_files = data_files
        cfg.stats_npz_path = str(results_dir / "stats.npz")
        cfg.event_bus_path = str(results_dir / "events.jsonl")

        return cfg


class SweepMetadataWriter:
    """Writes immutable metadata.json for a completed sweep."""

    def __init__(self, *, runner: str) -> None:
        self._runner = runner

    @staticmethod
    def _read_pyproject_project_info(pyproject_path: Path) -> tuple[str | None, str | None]:
        """Read [project] name/version from pyproject.toml.

        This is used as a fallback when the project is executed from source without
        being installed as a distribution (importlib.metadata won't find it).
        """

        try:
            raw = pyproject_path.read_bytes()
        except OSError:
            return (None, None)

        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            return (None, None)

        if "project" not in data:
            return (None, None)

        project = data["project"]
        name = project["name"] if isinstance(project, dict) and "name" in project else None
        version = project["version"] if isinstance(project, dict) and "version" in project else None

        if not isinstance(name, str):
            name = None
        if not isinstance(version, str):
            version = None

        return (name, version)

    @staticmethod
    def _guess_repo_root(start: Path) -> Path | None:
        """Walk upwards until pyproject.toml is found."""

        current = start
        for _ in range(20):
            candidate = current / "pyproject.toml"
            if candidate.exists():
                return current
            if current.parent == current:
                return None
            current = current.parent
        return None

    @classmethod
    def _resolve_project_metadata(cls) -> dict[str, str | None]:
        """Resolve project name/version without failing the sweep."""

        repo_root = cls._guess_repo_root(Path(__file__).resolve())
        pyproject_path = (repo_root / "pyproject.toml") if repo_root is not None else None

        name_from_pyproject: str | None = None
        version_from_pyproject: str | None = None

        if pyproject_path is not None:
            name_from_pyproject, version_from_pyproject = cls._read_pyproject_project_info(
                pyproject_path
            )

        distribution_name = name_from_pyproject or "trading-platform"

        version: str | None
        source: str
        try:
            version = importlib.metadata.version(distribution_name)
            source = "importlib.metadata"
        except importlib.metadata.PackageNotFoundError:
            version = version_from_pyproject
            source = "pyproject.toml" if version is not None else "unknown"

        return {
            "name": distribution_name,
            "version": version,
            "source": source,
        }

    def write(
        self,
        *,
        ctx: SweepContext,
        status: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        duration_seconds = (finished_at - started_at).total_seconds()

        project_meta = self._resolve_project_metadata()

        metadata = {
            "schema_version": "1.0",
            "identity": {
                "experiment_id": ctx.experiment_id,
                "segment_id": ctx.segment_id,
                "sweep_id": ctx.sweep_id,
            },
            "lifecycle": {
                "status": status,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": duration_seconds,
                "runner": self._runner,
            },
            "parameters": ctx.parameters,
            "code": {
                "git": {
                    "commit": os.environ.get("GIT_COMMIT"),
                    "dirty": os.environ.get("GIT_DIRTY") == "1",
                    "branch": os.environ.get("GIT_BRANCH"),
                },
                "project": {
                    "name": project_meta["name"],
                    "version": project_meta["version"],
                    "version_source": project_meta["source"],
                },
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "container_image": os.environ.get("IMAGE_TAG"),
            },
            "artifacts": {
                "stats": "stats.npz",
                "events": "events.jsonl",
            },
            "links": {},
        }

        target = ctx.scratch_results_dir / "sweep_metadata.json"
        target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class SweepResultPersister:
    """
    Persists sweep results from scratch to S3.

    Upload is atomic at sweep level via a _DONE marker.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "backtests",
    ) -> None:
        self._s3 = OCIObjectStorageS3Shim(region="eu-frankfurt-1")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def persist(self, ctx: SweepContext) -> None:
        results_dir = ctx.scratch_results_dir
        done_marker = results_dir / "_DONE"

        if not results_dir.exists():
            raise RuntimeError(f"Results directory does not exist: {results_dir}")

        if not done_marker.exists():
            raise RuntimeError(
                f"Sweep results not finalized (_DONE missing): {results_dir}"
            )

        s3_base = self._s3_base_scratch_prefix(ctx)

        for path in results_dir.iterdir():
            if path.is_dir():
                continue

            key = f"{s3_base}/{path.name}"
            self._upload_file(path, key)

    def _upload_file(self, path: Path, key: str) -> None:
        with path.open("rb") as fh:
            self._s3.put_object(
                bucket=self._bucket,
                key=key,
                body=fh,
            )

    def _s3_base_scratch_prefix(self, ctx: SweepContext) -> str:
        return (
            f"{self._prefix}/"
            f"{ctx.experiment_id}/"
            f"{ctx.segment_id}/"
            f"{ctx.sweep_id}"
        )


class SweepCleaner:
    """
    Handles safe cleanup of sweep scratch directories.

    Invariant:
    - Only sweep-private state may be removed during parallel execution.
    - Segment-level directories are shared across sweeps and must not be
      deleted by a single sweep.

    Cleanup is allowed ONLY after successful persistence.
    """

    def __init__(self, *, keep_scratch: bool) -> None:
        self._keep_scratch = keep_scratch

    def cleanup(self, ctx: SweepContext) -> None:
        """
        Remove the sweep's private scratch subtree.

        This deletes only:

            <scratch_root>/<experiment>/<segment>/results/<sweep_id>/

        It intentionally does NOT delete the segment directory itself, since that
        directory is shared by all sweeps in the segment (parallel execution).
        """
        if self._keep_scratch:
            return

        sweep_results_dir = ctx.scratch_results_dir
        if not sweep_results_dir.exists():
            return

        self._validate_target(ctx, sweep_results_dir)
        shutil.rmtree(sweep_results_dir)

    @staticmethod
    def _validate_target(ctx: SweepContext, target_dir: Path) -> None:
        """
        Guard rails against accidental deletion of shared directories.

        This method raises if the computed target does not match the expected
        sweep results layout.
        """
        if target_dir.name != ctx.sweep_id:
            raise RuntimeError(
                "Refusing to delete: target_dir does not match sweep_id "
                f"({target_dir} vs {ctx.sweep_id})"
            )

        if target_dir.parent.name != "results":
            raise RuntimeError(
                "Refusing to delete: target_dir is not under a 'results' folder "
                f"({target_dir})"
            )

        segment_dir = ctx.scratch_segment_dir
        try:
            resolved_target = target_dir.resolve()
            resolved_segment = segment_dir.resolve()
        except FileNotFoundError:
            # If a parent directory was removed concurrently, treat as no-op.
            return

        if not resolved_target.is_relative_to(resolved_segment):
            raise RuntimeError(
                "Refusing to delete: target_dir is outside scratch_segment_dir "
                f"({resolved_target} not under {resolved_segment})"
            )


def main() -> None:
    parser = argparse.ArgumentParser("run single backtest sweep")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load sweep context
    # ------------------------------------------------------------------

    if not args.context.exists():
        raise FileNotFoundError(
            f"SweepContext file does not exist: {args.context}. "
            "Ensure it is mounted as an Argo artifact."
        )

    ctx = SweepContext(**json.loads(args.context.read_text(encoding="utf-8")))
    ctx = replace(ctx, scratch_root=args.scratch_root)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    materializer = SweepMaterializer(bucket="data")
    materializer.materialize(ctx)

    engine_cfg = HftEngineConfig(**ctx.parameters["engine"])
    strategy_cfg = StrategyConfig(**ctx.parameters["strategy"])
    risk_cfg = RiskConfig(**ctx.parameters["risk"])

    runner = SweepEngineRunner(
        engine_cfg=engine_cfg,
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
    )

    persister = SweepResultPersister(bucket="data")

    metadata_writer = SweepMetadataWriter(runner="argo")
    cleaner = SweepCleaner(keep_scratch=False)

    # ------------------------------------------------------------------
    # Execute sweep
    # ------------------------------------------------------------------

    started_at = datetime.now(timezone.utc)
    status = "success"

    try:
        print(runner.run(ctx))
    except Exception:
        status = "failed"
        raise
    else:
        finished_at = datetime.now(timezone.utc)

        # Metadata is ALWAYS written
        metadata_writer.write(
            ctx=ctx,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
        )

        # Persist results ONLY on success
        persister.persist(ctx)
    finally:
        # Sweep-level cleanup is ALWAYS allowed
        cleaner.cleanup(ctx)


if __name__ == "__main__":
    main()
