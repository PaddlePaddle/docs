.. _cn_api_fluid_Tensor:

Tensor
-------------------------------

.. py:function:: paddle.fluid.Tensor




Tensor用于表示多维张量，可以通过 ``np.array(tensor)`` 方法转换为numpy.ndarray。

**示例代码**

.. code-block:: python

      import paddle.fluid as fluid

      t = fluid.Tensor()

.. py:method::  set(array, place, zero_copy=False)

该接口根据输入的numpy array和设备place，设置Tensor的数据。

参数：
    - **array** (numpy.ndarray) - 要设置的numpy array，支持的数据类型为bool, float32, float64, int8, int32, int64, uint8, uint16。
    - **place** (CPUPlace|CUDAPlace|CUDAPinnedPlace) - 要设置的Tensor所在的设备。
    - **zero_copy** (bool，可选) - 是否与输入的numpy数组共享内存。此参数仅适用于CPUPlace。默认值为False。

返回：无。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            t = fluid.Tensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())

.. py:method::  shape(self: paddle.fluid.core_avx.Tensor) → List[int]

该接口返回Tensor的shape。

返回：Tensor的shape。

返回类型：List[int] 。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            t = fluid.Tensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            print(t.shape())  # [5, 30]