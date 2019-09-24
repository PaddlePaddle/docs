.. _cn_api_fluid_LoDTensorArray:

LoDTensorArray
-------------------------------

.. py:class:: paddle.fluid.LoDTensorArray

LoDTensorArray是由LoDTensor组成的数组，支持"[]"运算符、len()函数和for迭代等。

**示例代码**

.. code-block:: python
        
        import paddle.fluid as fluid
     
        arr = fluid.LoDTensorArray()   



.. py:method:: append(self: paddle.fluid.core_avx.LoDTensorArray, tensor: paddle.fluid.core.LoDTensor) → None

该接口将LoDTensor追加到LoDTensorArray后。

参数：
  - **tensor** (LoDTensor) - 追加的LoDTensor。

返回：无。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
     
            arr = fluid.LoDTensorArray()
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            arr.append(t)





