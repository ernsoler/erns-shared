import boto3
from botocore.exceptions import ClientError
from typing import Iterator


class S3Client:
    def __init__(self) -> None:
        self._client = boto3.client("s3")

    def upload(
        self,
        bucket: str,
        key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self._client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

    def download(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()  # type: ignore[no-any-return]

    def delete(self, bucket: str, key: str) -> None:
        self._client.delete_object(Bucket=bucket, Key=key)

    def key_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def list_keys(self, bucket: str, prefix: str = "") -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def presigned_get_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        return self._client.generate_presigned_url(  # type: ignore[no-any-return]
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )

    def presigned_put_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        return self._client.generate_presigned_url(  # type: ignore[no-any-return]
            "put_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
