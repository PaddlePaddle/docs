.. _cn_api_paddle_utils_deprecated

paddle_utils_deprecated
-------------------------------

.. py:function:: paddle.utils.deprecated(update_to="", since="", reason="")

对于即将废弃的API可以加入该装饰器，在调用对应 PaddlePaddle API 时，可以做如下两件事情：

  - 修改被装饰API的相关docstring，添加即将废弃警告。
  - 当相关API被调用时，向控制台输出相关warning信息 :class:`~exceptions.DeprecatedWarning`。

参数：

  - **since** (str) - 即将废弃相对应的版本号。
  - **update_to**  (str) - 新的API名称。
  - **reason** (str) - 即将废弃该API的原因。

返回：装饰器(装饰器函数或者装饰器类)。
 
