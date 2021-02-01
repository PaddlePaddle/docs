.. _cn_api_fluid_XPUPlace:

XPUPlace
-------------------------------

.. py:class:: paddle.XPUPlace





``XPUPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 或 ``LoDTensor`` 的 Baidu Kunlun XPU 设备。
每个 ``XPUPlace`` 有一个 ``dev_id`` （设备id）来表明当前的 ``XPUPlace`` 所代表的显卡编号，编号从 0 开始。
``dev_id`` 不同的 ``XPUPlace`` 所对应的内存不可相互访问。
这里编号指的是可见显卡的逻辑编号，而不是显卡实际的编号。
可以通过 ``XPU_VISIBLE_DEVICES`` 环境变量限制程序能够使用的 Baidu Kunlun XPU 设备，程序启动时会遍历当前的可见设备，并从 0 开始为这些设备编号。
如果没有设置 ``XPU_VISIBLE_DEVICES``，则默认所有的设备都是可见的，此时逻辑编号与实际编号是相同的。

参数：
  - **id** (int，可选) - XPU的设备ID。如果为 ``None``，则默认会使用 id 为 0 的设备。默认值为 ``None``。

**代码示例**

.. code-block:: python

       import paddle

       place = paddle.XPUPlace(0)




