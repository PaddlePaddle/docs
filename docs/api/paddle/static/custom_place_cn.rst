.. _cn_api_fluid_custom_places:

custom_places
-------------------------------

.. py:function:: paddle.static.custom_places(device_type, device_ids=None)


该接口根据 ``device_type`` 和 ``device_ids`` 创建一个或多个 ``paddle.CustomPlace`` 对象，并返回所创建的对象列表。

参数
:::::::::
  - **device_type** (str) - 自定义设备类型
  - **device_ids** (list(int)|tuple(int)，可选) - 自定义设备ID列表或元组。默认值为 ``None``。

返回
:::::::::
list[paddle.CustomPlace]，创建的 ``paddle.CustomPlace`` 列表。

代码示例
:::::::::
COPY-FROM: paddle.static.custom_places
