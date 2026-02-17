from __future__ import annotations

import json

from trading_platform.backtest.io.s3_adapter import OCIObjectStorageS3Shim
from trading_platform.backtest.orchestrator.manifest import DataFileMeta, DatasetManifest


class S3DatasetManifest(DatasetManifest):
    """
    DatasetManifest implementation backed by S3.

    Semantics:
    - Manifests live under a canonical prefix (e.g. s3://data/canonical/)
    - All filtering is semantic (venue, datatype, symbol, time)
    - Path layout is NOT part of the contract
    """

    def __init__(
        self,
        *,
        bucket: str,
        stage: str,
    ) -> None:
        self._s3 = OCIObjectStorageS3Shim(region="eu-frankfurt-1")
        self._bucket = bucket
        self._prefix = stage.rstrip("/")

    # ------------------------------------------------------------------

    def iter_files(
        self,
        *,
        start_ts_ns: int,
        end_ts_ns: int,
        symbol: str,
        venue: str,
        datatype: str,
    ) -> list[DataFileMeta]:
        files: list[DataFileMeta] = []

        for key in self._list_manifest_keys():
            manifest = self._load_manifest(key)

            dataset = manifest["dataset"]

            if dataset["venue"] != venue:
                continue

            if dataset["datatype"] != datatype:
                continue

            time_range = manifest["time_range_ns"]
            if not self._overlaps(
                start_ts_ns,
                end_ts_ns,
                time_range["start"],
                time_range["end"],
            ):
                continue

            for entry in manifest["files"]:
                if not self._overlaps(
                    start_ts_ns,
                    end_ts_ns,
                    entry["start_ts_ns"],
                    entry["end_ts_ns"],
                ):
                    continue

                manifest_key = key
                manifest_dir = manifest_key.rsplit("/", 1)[0]
                object_key = f"{manifest_dir}/{entry['file_id']}"

                files.append(
                    DataFileMeta(
                        file_id=entry["file_id"],
                        object_key=object_key,
                        start_ts_ns=entry["start_ts_ns"],
                        end_ts_ns=entry["end_ts_ns"],
                        size_bytes=entry["size_bytes"],
                        symbol=symbol,
                        venue=venue,
                        datatype=datatype,
                    )
                )

        return files

    # ------------------------------------------------------------------

    def _list_manifest_keys(self) -> list[str]:
        resp = self._s3.list_objects(
            bucket=self._bucket,
            prefix=self._prefix,
        )

        contents = resp.get("Contents", [])

        return [
            obj["Key"]
            for obj in contents
            if obj["Key"].endswith("/manifest.json")
        ]

    def _load_manifest(self, key: str) -> dict:
        resp = self._s3.get_object(
            bucket=self._bucket,
            key=key,
        )

        body = resp["Body"]

        if hasattr(body, "read"):
            raw_bytes = body.read()
        else:
            raw_bytes = body

        return json.loads(raw_bytes)

    @staticmethod
    def _overlaps(
        a_start: int,
        a_end: int,
        b_start: int,
        b_end: int,
    ) -> bool:
        return a_start < b_end and b_start < a_end
