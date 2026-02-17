from __future__ import annotations

import io
from pathlib import Path

from oci.auth.signers import InstancePrincipalsSecurityTokenSigner
from oci.config import from_file
from oci.object_storage import ObjectStorageClient
from oci.signer import Signer


class OCIObjectStorageS3Shim:
    """
    Lightweight adapter that exposes a small, S3-like interface on top of
    Oracle Cloud Infrastructure (OCI) Object Storage.

    The goal of this class is *API shape compatibility*, not feature parity:
    it mimics a minimal subset of the boto3 S3 client that is sufficient for
    simple readers/writers and data pipelines.

    Authentication modes:
      - "instance_principal":
          Uses the OCI Instance Principal of the current Compute instance.
          Suitable only when running on OCI infrastructure.
      - "api_key":
          Uses a user-scoped OCI API key (private PEM key + config file).
          Suitable for local development, CI, and non-OCI environments.

    Implemented operations:
      - put_object: upload an object (write)
      - list_objects: list objects under a bucket/prefix (read)
      - get_object: download an object (read)

    Design notes:
      - Method signatures and return shapes are intentionally boto3-like.
      - This adapter talks directly to OCI Object Storage APIs, NOT to the
        S3-compatibility HTTP endpoint.
      - Authorization is fully governed by OCI IAM policies.
    """
    def __init__(
        self,
        *,
        region: str | None = None,
        auth_mode: str = "instance_principal",
        oci_config_file: str | None = None,
        oci_profile: str = "DEFAULT",
    ) -> None:
        """
        Create a new Object Storage client wrapper.

        Parameters:
          region:
            OCI region identifier (e.g. "eu-frankfurt-1").
            If provided, it overrides the region in the OCI config file.

          auth_mode:
            Authentication strategy to use:
              - "instance_principal": use the instances identity (OCI-only)
              - "api_key": use a user API key defined in an OCI config file

          oci_config_file:
            Path to an OCI CLI-style config file (required for api_key auth).
            Typically "~/.oci/config".

          oci_profile:
            Profile name inside the OCI config file to load credentials from.
        """
        if auth_mode == "instance_principal":
            signer = InstancePrincipalsSecurityTokenSigner()
            config = {}

        elif auth_mode == "api_key":
            if oci_config_file is None:
                raise ValueError("oci_config_file is required for api_key auth")

            config = from_file(
                file_location=oci_config_file,
                profile_name=oci_profile,
            )
            signer = Signer(
                tenancy=config["tenancy"],
                user=config["user"],
                fingerprint=config["fingerprint"],
                private_key_file_location=config["key_file"],
                pass_phrase=config.get("pass_phrase"),
            )

        else:
            raise ValueError(f"Unknown auth_mode: {auth_mode}")

        client_kwargs = {}
        if region:
            client_kwargs["region"] = region

        self.client = ObjectStorageClient(
            config=config,
            signer=signer,
            **client_kwargs,
        )

        self.namespace = self.client.get_namespace().data

    def put_object(self, bucket: str, key: str, body, content_type: str = "application/octet-stream"):
        """
        Upload an object to an OCI Object Storage bucket.

        Parameters mirror boto3 semantics:
          Bucket: bucket name
          Key: object name (path-like)
          Body: bytes or file-like object
          ContentType: optional MIME type

        Returns a minimal boto3-like dict containing the object's ETag
        (if provided by OCI).
        """
        resp = self.client.put_object(
            namespace_name=self.namespace,
            bucket_name=bucket,
            object_name=key,
            put_object_body=body,
            content_type=content_type,
        )
        etag = None
        try:
            etag = resp.headers.get("etag")
        except Exception:
            pass
        return {"ETag": etag}

    def list_objects(
        self,
        bucket: str,
        prefix: str | None = None,
        continuation_token: str | None = None,
        max_keys: int = 1000,
    ) -> dict[str, object]:
        """
        List objects in a bucket, optionally filtered by prefix.

        This method approximates boto3's list_objects behavior:
          - 'Prefix' filters object names
          - pagination is exposed via ContinuationToken / NextContinuationToken

        Internally, this maps to OCI's 'list_objects' API, using:
          - 'prefix' for filtering
          - 'start' for pagination

        Returns:
          A dict with keys:
            - Contents: list of {"Key", "Size"}
            - IsTruncated: whether more results are available
            - NextContinuationToken: token for the next page (or None)
        """
        kwargs = {
            "namespace_name": self.namespace,
            "bucket_name": bucket,
            "limit": max_keys,
        }
        if prefix:
            kwargs["prefix"] = prefix
        if continuation_token:
            kwargs["start"] = continuation_token

        resp = self.client.list_objects(**kwargs)
        objects = []
        for o in resp.data.objects or []:
            objects.append({"Key": o.name, "Size": getattr(o, "size", None)})

        next_token = getattr(resp.data, "next_start_with", None)
        return {
            "Contents": objects,
            "IsTruncated": bool(next_token),
            "NextContinuationToken": next_token,
        }

    def get_object(self, bucket: str, key: str) -> dict[str, object]:
        """
        Download an object from OCI Object Storage.

        Returns a boto3-like response where:
          - 'Body' is a file-like object (io.BytesIO)
          - 'ContentLength' and 'ContentType' are best-effort metadata

        The OCI Python SDK exposes response bodies in different shapes
        depending on transport and SDK version; this method normalizes
        them into a single bytes buffer.
        """
        resp = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=bucket,
            object_name=key,
        )

        data_bytes = None
        d = resp.data

        # Case 1: direct .read()
        if hasattr(d, "read") and callable(getattr(d, "read")):
            data_bytes = d.read()

        # Case 2: .content (bytes already)
        elif hasattr(d, "content"):
            data_bytes = d.content

        # Case 3: raw.read()
        elif hasattr(d, "raw") and hasattr(d.raw, "read") and callable(getattr(d.raw, "read")):
            data_bytes = d.raw.read()

        # Case 4: stream chunks (fallback)
        elif hasattr(d, "raw") and hasattr(d.raw, "stream") and callable(getattr(d.raw, "stream")):
            chunks = []
            for chunk in d.raw.stream(1024 * 1024, decode_content=False):
                chunks.append(chunk)
            data_bytes = b"".join(chunks)

        else:
            raise TypeError("Unsupported OCI get_object response type; no readable data attribute found.")

        # Content-Length/Type (best effort)
        content_length = None
        try:
            content_length = int(resp.headers.get("content-length", "0"))
        except Exception:
            pass
        if not content_length and data_bytes is not None:
            content_length = len(data_bytes)

        content_type = ""
        try:
            content_type = resp.headers.get("content-type", "")
        except Exception:
            pass

        return {
            "Body": io.BytesIO(data_bytes if data_bytes is not None else b""),
            "ContentLength": content_length or 0,
            "ContentType": content_type,
        }

    def download_to_file(
        self,
        bucket: str,
        key: str,
        destination: str | Path,
        *,
        chunk_size_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        """
        Stream an object from OCI Object Storage directly to a local file.

        This method performs a chunked download over HTTPS and writes each
        chunk incrementally to disk. The entire object is never loaded into
        memory at once, ensuring constant and predictable RAM usage.

        Parameters:
            bucket:
                Name of the OCI Object Storage bucket.
            key:
                Object name (path-like key) within the bucket.
            destination:
                Local filesystem path where the object will be written.
                Parent directories must already exist.
            chunk_size_bytes:
                Size of each streamed chunk in bytes. Defaults to 8 MiB.

        Raises:
            RuntimeError:
                If the OCI response does not expose a streamable body.
        """
        destination_path = Path(destination)

        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=bucket,
            object_name=key,
        )

        data = response.data

        if not hasattr(data, "raw") or not hasattr(data.raw, "stream"):
            raise RuntimeError(
                "OCI get_object response does not expose a streamable body."
            )

        with destination_path.open("wb") as file_handle:
            for chunk in data.raw.stream(
                chunk_size_bytes,
                decode_content=False,
            ):
                file_handle.write(chunk)
