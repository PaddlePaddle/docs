.. _cn_api_fluid_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.utils.unique_name.switch(new_generator=None)




将当前上下文的命名空间切换到新的命名空间。该接口与 guard 接口都可用于更改命名空间，推荐使用 guard 接口，配合 with 语句管理命名空间上下文。

参数
::::::::::::

  - **new_generator** (UniqueNameGenerator，可选) - 要切换到的新命名空间，一般无需设置。缺省值为 None，表示切换到一个匿名的新命名空间。

返回
::::::::::::
UniqueNameGenerator，先前的命名空间，一般无需操作该返回值。

代码示例
::::::::::::

COPY-FROM: paddle.utils.unique_name.switch
