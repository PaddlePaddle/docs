.. _cn_api_paddle_NPUPlace:

NPUPlace
-------------------------------

.. py:class:: paddle.NPUPlace

``NPUPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 的 NPU 设备。
每个 ``NPUPlace`` 有一个 ``dev_id`` （设备 id）来表明当前的 ``NPUPlace`` 所代表的显卡编号，编号从 0 开始。
``dev_id`` 不同的 ``NPUPlace`` 所对应的内存不可相互访问。
这里编号指的是显卡实际的编号，而不是显卡的逻辑编号。

参数
::::::::::::

  - **id** (int，可选) - NPU 的设备 ID。

代码示例
::::::::::::

COPY-FROM: paddle.NPUPlace
