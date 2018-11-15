.. _cn_api_fluid_fluid_Tensor:

Tensor
:::::::

.. py:class:: paddle.fluid.Tensor
''''''''''''''''''''''''''''''''''''

    LoDTensor的别名


.. _cn_api_fluid_switch_scope:

_switch_scope
:::::::::::::::

.. py:class:: paddle.fluid._switch_scope(scope)
'''''''''''''''''''''''''''''''''''''''''''''''''


.. _cn_api_fluid_Scope:

Scope
::::::::

.. py:class:: paddle.fluid.scope(scope)
''''''''''''''''''''''''''''''''''''''''''''''''

.. py:method:: drop_kids(self: paddle.fluid.core.Scope) → None
.. py:method:: find_var(self: paddle.fluid.core.Scope, arg0: unicode) → paddle.fluid.core.Variable
.. py:method:: new_scope(self: paddle.fluid.core.Scope) → paddle.fluid.core.Scope
.. py:method:: var(self: paddle.fluid.core.Scope, arg0: unicode) → paddle.fluid.core.Variable   


.. _cn_api_fluid_LoDTensorArray:

LoDTensorArray
::::::::::::::::

.. py:class:: paddle.fluid.LoDTensorArray
''''''''''''''''''''''''''''''''''''''''''''''''

.. py:method:: append(self: paddle.fluid.core.LoDTensorArray, arg0: paddle.fluid.core.LoDTensor) → None



.. _cn_api_fluid_CPUPlace:

CPUPlace
::::::::::::::::

.. py:class:: paddle.fluid.CPUPlace
''''''''''''''''''''''''''''''''''''''''''''''''


.. _cn_api_fluid_CUDAPlace:

CUDAPlace
::::::::::::::::

.. py:class:: paddle.fluid.CUDAPlace
''''''''''''''''''''''''''''''''''''''''''''''''


.. _cn_api_fluid_CUDAPinnedPlace:

CUDAPinnedPlace
::::::::::::::::

.. py:class:: paddle.fluid.CUDAPinnedPlace
''''''''''''''''''''''''''''''''''''''''''''''''


.. _cn_api_fluid_CPUPlace:

CPUPlace
::::::::::::::::

.. py:class:: paddle.fluid.CPUPlace
''''''''''''''''''''''''''''''''''''''''''''''''




**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]
        
参数:   
    - **input（Variable）**: 作为索引序列的输入变量。
    - **win_size（int）**: 枚举所有子序列的窗口大小。
    - **pad_value（int）**: 填充值，默认为0。
          
返回:  枚举序列变量是LoD张量（LoDTensor）。
          
**代码示例**

..  code-block:: python

      x = fluid.layers.data(shape[30, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)
      
