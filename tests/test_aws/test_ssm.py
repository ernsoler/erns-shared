import pytest
import boto3
from moto import mock_aws

from erns_shared.aws.ssm import SSMClient


@pytest.fixture
def ssm(aws_credentials):
    with mock_aws():
        client = boto3.client("ssm", region_name="us-east-1")
        client.put_parameter(Name="/app/db_url", Value="postgres://localhost/db", Type="String")
        client.put_parameter(Name="/app/secret_key", Value="s3cr3t", Type="SecureString")
        client.put_parameter(Name="/app/nested/value", Value="nested-val", Type="String")
        yield SSMClient()


class TestGetParameter:
    def test_returns_value(self, ssm):
        assert ssm.get_parameter("/app/db_url", decrypt=False) == "postgres://localhost/db"

    def test_cache_hit_skips_api(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False)
        ssm._client = None  # break the client — cache must serve the value
        assert ssm.get_parameter("/app/db_url", decrypt=False) == "postgres://localhost/db"

    def test_bypass_cache_goes_to_api(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False, use_cache=True)
        val = ssm.get_parameter("/app/db_url", decrypt=False, use_cache=False)
        assert val == "postgres://localhost/db"

    def test_value_stored_in_cache(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False)
        assert "/app/db_url" in ssm._cache

    def test_bypass_cache_does_not_populate_cache(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False, use_cache=False)
        assert "/app/db_url" not in ssm._cache


class TestGetParametersByPath:
    def test_returns_all_parameters_under_path(self, ssm):
        params = ssm.get_parameters_by_path("/app", decrypt=False)
        assert "/app/db_url" in params
        assert "/app/nested/value" in params

    def test_values_are_correct(self, ssm):
        params = ssm.get_parameters_by_path("/app", decrypt=False)
        assert params["/app/db_url"] == "postgres://localhost/db"

    def test_cached_after_path_fetch(self, ssm):
        ssm.get_parameters_by_path("/app", decrypt=False)
        ssm._client = None  # break client — cache must serve subsequent gets
        assert ssm.get_parameter("/app/db_url", decrypt=False) == "postgres://localhost/db"


class TestPutParameter:
    def test_put_then_get(self, ssm):
        ssm.put_parameter("/app/new_key", "new_value", param_type="String")
        assert ssm.get_parameter("/app/new_key", decrypt=False) == "new_value"

    def test_put_clears_stale_cache_entry(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False)
        ssm.put_parameter("/app/db_url", "updated_url", param_type="String", overwrite=True)
        assert "/app/db_url" not in ssm._cache

    def test_put_with_overwrite(self, ssm):
        ssm.put_parameter("/app/db_url", "new_url", param_type="String", overwrite=True)
        assert ssm.get_parameter("/app/db_url", decrypt=False) == "new_url"


class TestInvalidateCache:
    def test_invalidate_single_key(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False)
        ssm.get_parameter("/app/nested/value", decrypt=False)
        ssm.invalidate_cache("/app/db_url")
        assert "/app/db_url" not in ssm._cache
        assert "/app/nested/value" in ssm._cache

    def test_invalidate_all(self, ssm):
        ssm.get_parameter("/app/db_url", decrypt=False)
        ssm.get_parameter("/app/nested/value", decrypt=False)
        ssm.invalidate_cache()
        assert ssm._cache == {}

    def test_invalidate_nonexistent_key_is_noop(self, ssm):
        ssm.invalidate_cache("/does/not/exist")  # should not raise
