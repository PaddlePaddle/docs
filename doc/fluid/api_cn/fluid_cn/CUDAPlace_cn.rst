.. _cn_api_fluid_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPlace




.. note::
    多卡任务请先使用 FLAGS_selected_gpus 环境变量设置可见的GPU设备，下个版本将会修正 CUDA_VISIBLE_DEVICES 环境变量无效的问题。

``CUDAPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 或 ``LoDTensor`` 的 GPU 设备。
每个 ``CUDAPlace`` 有一个 ``dev_id`` （设备id）来表明当前的 ``CUDAPlace`` 所代表的显卡编号，编号从 0 开始。
``dev_id`` 不同的 ``CUDAPlace`` 所对应的内存不可相互访问。
这里编号指的是可见显卡的逻辑编号，而不是显卡实际的编号。
可以通过 ``CUDA_VISIBLE_DEVICES`` 环境变量限制程序能够使用的 GPU 设备，程序启动时会遍历当前的可见设备，并从 0 开始为这些设备编号。
如果没有设置 ``CUDA_VISIBLE_DEVICES``，则默认所有的设备都是可见的，此时逻辑编号与实际编号是相同的。

参数：
  - **id** (int，可选) - GPU的设备ID。如果为 ``None``，则默认会使用 id 为 0 的设备。默认值为 ``None``。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       gpu_place = fluid.CUDAPlace(0)




