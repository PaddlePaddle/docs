.. _cn_api_paddle_utils_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.utils.unique_name.switch(new_generator=None, new_para_name_checker=None)




将当前上下文的命名空间切换到新的命名空间。该接口与 guard 接口都可用于更改命名空间，推荐使用 guard 接口，配合 with 语句管理命名空间上下文。

参数
::::::::::::

    - **new_generator** (UniqueNameGenerator，可选) - 要切换到的新命名空间，一般无需设置。缺省值为 None，表示切换到一个匿名的新命名空间。
    - **new_para_name_checker** (DygraphParameterNameChecker，可选) - 要切换到的参数名称检查器，一般无需设置。缺省值为 None，表示切换到新的参数名称检查器。

返回
::::::::::::
tuple[UniqueNameGenerator, DygraphParameterNameChecker]，分别为先前的命名空间和参数名称检查器，一般无需操作该返回值。

代码示例
::::::::::::

COPY-FROM: paddle.utils.unique_name.switch
