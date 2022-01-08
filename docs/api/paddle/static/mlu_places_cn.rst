.. _cn_api_fluid_mlu_places:

mlu_places
-------------------------------

.. py:function:: paddle.static.mlu_places(device_ids=None)


.. note::
    多卡任务请先使用 FLAGS_selected_mlus 环境变量设置可见的MLU设备。

该接口根据 ``device_ids`` 创建一个或多个 ``paddle.device.MLUPlace`` 对象，并返回所创建的对象列表。

如果 ``device_ids`` 为 ``None``，则首先检查 ``FLAGS_selected_mlus`` 标志。
例如： ``FLAGS_selected_mlus=0,1,2`` ，则返回的列表将为 ``[paddle.device.MLUPlace(0), paddle.device.MLUPlace(1), paddle.device.MLUPlace(2)]``。
如果未设置标志 ``FLAGS_selected_mlus`` ，则返回所有可见的 MLU places。

如果 ``device_ids`` 不是 ``None``，它应该是使用的MLU设备ID的列表或元组。
例如： ``device_id=[0,1,2]`` ，返回的列表将是 ``[paddle.device.MLUPlace(0), paddle.device.MLUPlace(1), paddle.device.MLUPlace(2)]``。

参数
:::::::::
  - **device_ids** (list(int)|tuple(int)，可选) - MLU的设备ID列表或元组。默认值为 ``None``。

返回
:::::::::
list[paddle.device.MLUPlace]，创建的 ``paddle.device.MLUPlace`` 列表。

代码示例
:::::::::
COPY-FROM: paddle.static.mlu_places

