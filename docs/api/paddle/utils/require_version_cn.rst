.. _cn_api_paddle_utils_require_version:

paddle.utils.require_version
---------------------

.. py:function:: paddle.utils.require_version(min_version, max_version=None)

检查 PaddlePaddle 的安装版本是否在区间[min_version，max_version]内，如果安装版本低于 min_version 或高于 max_version ，将抛出异常，如果安装版本满足则没有返回值。

参数
::::::::::::
- **min_version** (str) - 所需的最低版本（如 '1.4.0'）。
- **max_version** (str, 可选) - 所需的最大版本（如 '1.6.0'），默认为 None，这意味着任何等于或高于 min_version 的版本都是可接受的。

返回
::::::::::::
None（满足版本要求是没有返回值的）

异常
::::::::::::

- **TypeError** – 如果 min_version 不是 str 类型。

- **TypeError** – 如皋 max_version 不是 str 类型或者 None。

- **ValueError** – 如果 min_version 不是有效的版本格式。

- **ValueError** – 如果 max_version 不是有效的版本格式或者 None。

- **Exception** – 如果已安装的版本不在 min_version 和 max_version 之间。

代码示例
::::::::::::

COPY-FROM: paddle.utils.require_version
