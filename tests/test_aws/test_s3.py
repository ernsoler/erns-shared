import pytest
import boto3
from moto import mock_aws

from erns_shared.aws.s3 import S3Client

_BUCKET = "test-bucket"
_KEY = "folder/file.txt"
_BODY = b"hello world"


@pytest.fixture
def s3(aws_credentials):
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket=_BUCKET)
        yield S3Client()


class TestUploadDownload:
    def test_upload_and_download_roundtrip(self, s3):
        s3.upload(bucket=_BUCKET, key=_KEY, body=_BODY)
        assert s3.download(bucket=_BUCKET, key=_KEY) == _BODY

    def test_upload_sets_content_type(self, s3):
        s3.upload(bucket=_BUCKET, key=_KEY, body=_BODY, content_type="text/plain")
        meta = boto3.client("s3").head_object(Bucket=_BUCKET, Key=_KEY)
        assert meta["ContentType"] == "text/plain"

    def test_delete_removes_key(self, s3):
        s3.upload(bucket=_BUCKET, key=_KEY, body=_BODY)
        s3.delete(bucket=_BUCKET, key=_KEY)
        assert not s3.key_exists(bucket=_BUCKET, key=_KEY)


class TestKeyExists:
    def test_returns_true_when_present(self, s3):
        s3.upload(bucket=_BUCKET, key=_KEY, body=_BODY)
        assert s3.key_exists(bucket=_BUCKET, key=_KEY) is True

    def test_returns_false_when_absent(self, s3):
        assert s3.key_exists(bucket=_BUCKET, key="nonexistent.txt") is False


class TestListKeys:
    def test_lists_uploaded_keys(self, s3):
        keys = ["a.txt", "b.txt", "c.txt"]
        for k in keys:
            s3.upload(bucket=_BUCKET, key=k, body=b"x")
        assert sorted(s3.list_keys(bucket=_BUCKET)) == sorted(keys)

    def test_prefix_filter(self, s3):
        s3.upload(bucket=_BUCKET, key="docs/a.txt", body=b"x")
        s3.upload(bucket=_BUCKET, key="imgs/b.png", body=b"x")
        assert list(s3.list_keys(bucket=_BUCKET, prefix="docs/")) == ["docs/a.txt"]

    def test_empty_bucket_returns_nothing(self, s3):
        assert list(s3.list_keys(bucket=_BUCKET)) == []


class TestPresignedUrls:
    def test_presigned_get_url_returns_string(self, s3):
        s3.upload(bucket=_BUCKET, key=_KEY, body=_BODY)
        url = s3.presigned_get_url(bucket=_BUCKET, key=_KEY)
        assert isinstance(url, str) and _BUCKET in url

    def test_presigned_put_url_returns_string(self, s3):
        url = s3.presigned_put_url(bucket=_BUCKET, key=_KEY)
        assert isinstance(url, str) and _BUCKET in url

    def test_presigned_url_custom_expiration_accepted(self, s3):
        url = s3.presigned_get_url(bucket=_BUCKET, key=_KEY, expiration=60)
        assert isinstance(url, str) and _BUCKET in url
