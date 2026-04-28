import boto3
from typing import Dict, Optional


class SSMClient:
    """SSM Parameter Store client with in-process cache."""

    def __init__(self) -> None:
        self._client = boto3.client("ssm")
        self._cache: Dict[str, str] = {}

    def get_parameter(self, name: str, decrypt: bool = True, use_cache: bool = True) -> str:
        if use_cache and name in self._cache:
            return self._cache[name]
        response = self._client.get_parameter(Name=name, WithDecryption=decrypt)
        value: str = response["Parameter"]["Value"]
        if use_cache:
            self._cache[name] = value
        return value

    def get_parameters_by_path(
        self, path: str, decrypt: bool = True, use_cache: bool = True
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        paginator = self._client.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path=path, WithDecryption=decrypt, Recursive=True):
            for param in page.get("Parameters", []):
                name: str = param["Name"]
                value: str = param["Value"]
                result[name] = value
                if use_cache:
                    self._cache[name] = value
        return result

    def put_parameter(
        self,
        name: str,
        value: str,
        param_type: str = "SecureString",
        overwrite: bool = False,
    ) -> None:
        self._client.put_parameter(
            Name=name, Value=value, Type=param_type, Overwrite=overwrite
        )
        self._cache.pop(name, None)

    def invalidate_cache(self, name: Optional[str] = None) -> None:
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()
