.. _cn_api_fluid_LoDTensorArray:

LoDTensorArray
-------------------------------

.. py:class:: paddle.fluid.LoDTensorArray

LoDTensor的数组。

**示例代码**

.. code-block:: python
        
        import paddle.fluid as fluid
     
        arr = fluid.LoDTensorArray()   

.. py:method:: append(self: paddle.fluid.core_avx.LoDTensorArray, tensor: paddle.fluid.core.LoDTensor) → None

将LoDTensor追加到LoDTensorArray后。

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
     
            arr = fluid.LoDTensorArray()
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            arr.append(t)





