.. _cn_api_paddle_hub_load_state_dict_from_url:

load_state_dict_from_url
-------------------------------

.. py:function:: paddle.hub.load_state_dict_from_url(url, model_dir=None, check_hash=False, file_name=None, map_location=None)

用于从指定的 URL 下载 Paddle 的模型权重（即 state_dict），并在必要时解压下载的文件。


参数
:::::::::

    - **url** (str) - 要下载对象的 URL 地址。若为 None，则默认保存到 <PADDLE_HOME>/hub/checkpoints，其中 <PADDLE_HOME> 为 PADDLE_HOME 环境变量的值，若未设置 PADDLE_HOME，则默认保存到 ~/.cache/paddle。
    - **model_dir** (str，可选) - 保存下载文件的目录。
    - **check_hash** (bool，可选) - 是否验证文件的 SHA256 哈希值。若为 True，则 URL 中的文件名部分需遵循命名约定 `filename-<sha256>.ext`，其中 `<sha256>` 为文件内容 SHA256 哈希值的前八位或更多，用于确保文件名唯一性并验证文件内容。默认值：False。
    - **file_name** (str，可选) - 下载文件的自定义文件名。如果未设置，将使用 URL 中的文件名。
    - **map_location** (str，可选) - 指定存储位置的映射方式。支持 `"cpu"`, `"gpu"`, `"xpu"`, `"npu"`, `"numpy"`, `"np"` 等。默认值：None。

返回
:::::::::

    返回可以在 Paddle 中使用的目标对象。

代码示例
:::::::::

COPY-FROM: paddle.hub.load_state_dict_from_url
