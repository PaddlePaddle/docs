.. _cn_api_paddle_utils_deprecated

paddle_utils_deprecated
-------------------------------

.. py:function:: paddle.utils.deprecated(update_to="", since="", reason="")

对于过时的API可以加入该装饰器，在调用对应 PaddlePaddle API 时，可以做如下两件事情：

  - 修改被装饰API的相关docstring，添加过时警告。
  - 当相关API被调用时，向控制台输出相关warning信息 :class:`~exceptions.DeprecatedWarning`。

参数：

  - **since** (str) - 下载的链接。
  - **update_to**  (str) - 下载的链接。
  - **reason** (str) - 下载的链接。

返回：装饰器(装饰器函数或者装饰器类)。
 
