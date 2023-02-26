.. _cn_api_paddle_utils_deprecated:

paddle_utils_deprecated
-------------------------------

.. py:function:: paddle.utils.deprecated(update_to="", since="", reason="", level=0)

对于即将废弃的 API 可以加入该装饰器，在调用对应 PaddlePaddle API 时，可以做如下两件事情：

  - 修改被装饰 API 的相关 docstring，添加即将废弃警告。
  - 当相关 API 被调用时，向控制台输出相关 warning 信息 :class:`~exceptions.DeprecatedWarning`。

参数
::::::::::::


  - **since** (str) - 即将废弃相对应的版本号。
  - **update_to**  (str) - 新的 API 名称。
  - **reason** (str) - 即将废弃该 API 的原因。
  - **level** (int) - 已弃用的警告日志级别。他必须是一个整数，而且是 0, 1, 2 中的一个。
                    如果' level == 0 '，则不显示警告信息。
                    如果' level == 1 '，则正常显示警告信息。
                    如果' level == 2 '，它将引发' RuntimeError '。

返回
::::::::::::
装饰器(装饰器函数或者装饰器类)。
