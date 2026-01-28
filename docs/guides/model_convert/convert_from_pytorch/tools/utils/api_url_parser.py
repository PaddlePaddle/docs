from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urljoin

from loguru import logger
from sphobjinv.inventory import Inventory


class ApiUrlParserBase(ABC):
    """
    Base Parser class for api url
    """

    @abstractmethod
    def get_api_url(self, api: str) -> str | None:
        """
        Parse the api url
        """
        raise NotImplementedError


class EmptyApiUrlParser(ApiUrlParserBase):
    def get_api_url(self, api: str) -> str | None:
        return None


class InventoryUrlParser(ApiUrlParserBase):
    def __init__(self, inventory_path_or_url: str, base_url: str) -> None:
        self._do_init(inventory_path_or_url, base_url)
        super().__init__()

    def _do_init(self, inventory_path_or_url: str, base_url: str):
        if "http" in inventory_path_or_url:
            inv = Inventory(url=inventory_path_or_url)  # type: ignore
        else:
            inv = Inventory(source=Path(inventory_path_or_url))  # type: ignore
        self._inv = {}
        for _, v in inv.json_dict(expand=True, contract=False).items():
            if isinstance(v, dict) and v.get("domain") == "py":
                if v["name"] in self._inv:
                    logger.warning(
                        f"Duplicated api {v['name']} found, already exists with {self._inv[v['name']]}, but new url is {urljoin(base_url, v['uri'])}"
                    )
                else:
                    self._inv[v["name"]] = urljoin(base_url, v["uri"])

    def get_api_url(self, api: str) -> str | None:
        """
        Find api name in "py" domain.
        """
        return self._inv.get(api, None)


class PaddleInventoryUrlParser(ApiUrlParserBase):
    def __init__(
        self, zh_url: str, zh_base_url: str, en_url: str, en_base_url: str
    ) -> None:
        self._do_init(zh_url, zh_base_url, en_url, en_base_url)
        super().__init__()

    def _do_init(
        self, zh_url: str, zh_base_url: str, en_url: str, en_base_url: str
    ):
        """
        Rules:
        1. zh docs priority to en, if not found in zh docs, fall back to en version.
        2. Only search "py" domain, except items start with "/api/paddle/tensor__upper_cn.rst"
           in "std" domain, for "paddle.Tensor.xxx" exists in these items.

        e.g.
          "10467": {
            "name": "/api/paddle/tensor__upper_cn.rst#byte",
            "domain": "std",
            "role": "label",
            "priority": "-1",
            "uri": "api/paddle/Tensor__upper_cn.html#byte",
            "dispname": "byte()",
            },
        """
        self._inv = {}

        def add_inv(inv, base_url, ignore_duplicate=False):
            for _, v in inv.json_dict(expand=True, contract=False).items():
                if not isinstance(v, dict):
                    continue
                name = v.get("name", "")
                domain = v.get("domain", "")
                if v.get("domain") == "py":
                    if v["name"] in self._inv:
                        if ignore_duplicate:
                            continue
                        raise ValueError(
                            f"Duplicated api name {v['name']} found"
                        )
                    else:
                        self._inv[name] = urljoin(base_url, v["uri"])
                elif domain == "std" and name.startswith(
                    "/api/paddle/tensor__upper_cn.rst"
                ):
                    dispname = v.get("dispname", "")
                    if dispname == "":
                        continue
                    api_name = "paddle.Tensor." + dispname.split("(")[0]
                    if (
                        api_name in self._inv
                        and api_name != "paddle.Tensor.Tensor"
                    ):
                        logger.warning(
                            f"Duplicated api {api_name} found, already exists with {self._inv[api_name]}, but new url is {urljoin(base_url, v['uri'])}"
                        )
                    else:
                        self._inv[api_name] = urljoin(base_url, v["uri"])

        inv_zh = Inventory(url=zh_url)  # type: ignore
        inv_en = Inventory(url=en_url)  # type: ignore

        add_inv(inv_zh, zh_base_url)
        add_inv(inv_en, en_base_url, ignore_duplicate=True)

    def get_api_url(self, api: str) -> str | None:
        return self._inv.get(api, None)


_parser = {}


def get_parser(name: str) -> ApiUrlParserBase:
    if name in _parser:
        return _parser[name]
    elif name in ("torch", "pytorch"):
        _parser[name] = InventoryUrlParser(
            "https://docs.pytorch.org/docs/stable/objects.inv",
            "https://pytorch.org/docs/stable/",
        )
        return _parser[name]
    elif name in ("paddle", "paddlepaddle"):
        _parser[name] = PaddleInventoryUrlParser(
            "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/objects.inv",
            "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/",
            "https://www.paddlepaddle.org.cn/documentation/docs/en/develop/objects.inv",
            "https://www.paddlepaddle.org.cn/documentation/docs/en/develop/",
        )
        return _parser[name]
    elif name == "torchvision":
        _parser[name] = InventoryUrlParser(
            "https://pytorch.org/vision/stable/objects.inv",
            "https://pytorch.org/vision/stable/",
        )
        return _parser[name]
    elif name == "transformers":
        _parser[name] = InventoryUrlParser(
            "https://huggingface.co/docs/transformers/main/en/objects.inv",
            "https://huggingface.co/docs/transformers/main/en/",
        )
        return _parser[name]
    elif name == "fairscale":
        _parser[name] = InventoryUrlParser(
            "https://fairscale.readthedocs.io/en/latest/objects.inv",
            "https://fairscale.readthedocs.io/en/latest/",
        )
        return _parser[name]
    elif name == "flash_attn":
        _parser[name] = EmptyApiUrlParser()
        return _parser[name]

    raise ValueError(f"Unknown name {name}")
