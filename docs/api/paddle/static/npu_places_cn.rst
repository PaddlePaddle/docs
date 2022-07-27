.. _cn_api_fluid_npu_places:

npu_places
-------------------------------

.. py:function:: paddle.static.npu_places(device_ids=None)


.. note::
    多卡任务请先使用 FLAGS_selected_npus 环境变量设置可见的 NPU 设备。

该接口根据 ``device_ids`` 创建一个或多个 ``paddle.NPUPlace`` 对象，并返回所创建的对象列表。

如果 ``device_ids`` 为 ``None``，则首先检查 ``FLAGS_selected_npus`` 标志。
例如：``FLAGS_selected_npus=0,1,2``，则返回的列表将为 ``[paddle.NPUPlace(0), paddle.NPUPlace(1), paddle.NPUPlace(2)]``。
如果未设置标志 ``FLAGS_selected_npus``，则返回所有可见的 NPU places。

如果 ``device_ids`` 不是 ``None``，它应该是使用的 NPU 设备 ID 的列表或元组。
例如：``device_id=[0,1,2]``，返回的列表将是 ``[paddle.NPUPlace(0), paddle.NPUPlace(1), paddle.NPUPlace(2)]``。

参数
:::::::::
  - **device_ids** (list(int)|tuple(int)，可选) - NPU 的设备 ID 列表或元组。默认值为 ``None``。

返回
:::::::::
list[paddle.NPUPlace]，创建的 ``paddle.NPUPlace`` 列表。

代码示例
:::::::::
COPY-FROM: paddle.static.npu_places
