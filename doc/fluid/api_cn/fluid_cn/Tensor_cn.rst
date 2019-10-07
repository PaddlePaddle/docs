.. _cn_api_fluid_Tensor:

Tensor
-------------------------------

.. py:function:: paddle.fluid.Tensor

Tensor用于表示多维张量，可以通过 ``np.array(tensor)`` 方法转换为numpy.ndarray。

**示例代码**

.. code-block:: python

      import paddle.fluid as fluid

      t = fluid.Tensor()

.. py:method::  set(*args, **kwargs)

该接口根据输入的numpy array和设备place，设置Tensor的数据。

重载函数：

1. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CPUPlace) -> None

2. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CPUPlace) -> None

3. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CPUPlace) -> None

4. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CPUPlace) -> None

5. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CPUPlace) -> None

6. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CPUPlace) -> None

7. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CPUPlace) -> None

8. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CPUPlace) -> None

9. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CUDAPlace) -> None

10. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CUDAPlace) -> None

11. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CUDAPlace) -> None

12. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CUDAPlace) -> None

13. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CUDAPlace) -> None

14. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CUDAPlace) -> None

15. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CUDAPlace) -> None

16. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CUDAPlace) -> None

17. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CUDAPinnedPlace) -> None

18. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CUDAPinnedPlace) -> None

19. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CUDAPinnedPlace) -> None

20. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CUDAPinnedPlace) -> None

21. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CUDAPinnedPlace) -> None

22. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CUDAPinnedPlace) -> None

23. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CUDAPinnedPlace) -> None

24. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CUDAPinnedPlace) -> None

参数：
    - **array** (numpy.ndarray) - 要设置的numpy array，支持的数据类型为bool, float32, float64, int8, int32, int64, uint8, uint16。
    - **place** (CPUPlace|CUDAPlace|CUDAPinnedPlace) - 要设置的Tensor所在的设备。

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