.. _cn_api_paddle_utils_deprecated:

deprecated
-------------------------------

.. py:function:: paddle.utils.deprecated(update_to="", since="", reason="", level=0)

对于即将废弃的 API 可以加入该装饰器，在调用对应 PaddlePaddle API 时，可以做如下两件事情：

  - 修改被装饰 API 的相关 docstring，添加即将废弃警告。
  - 当相关 API 被调用时，向控制台输出相关 warning 信息 :class:`~exceptions.DeprecatedWarning`。

参数
::::::::::::


  - **update_to**  (str，可选) - 新的 API 名称。
  - **since** (str，可选) - 即将废弃相对应的版本号。
  - **reason** (str，可选) - 即将废弃该 API 的原因。
  - **level** (int，可选) - 废弃警告的日志级别，只能为 0、1 或 2。为 0 时不显示警告信息；为 1 时正常显示警告信息；为 2 时抛出 ``RuntimeError``。默认值为 0。

返回
::::::::::::
装饰器(装饰器函数或者装饰器类)。
