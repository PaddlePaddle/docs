.. _cn_api_static_cn_InputSpec:

InputSpec
-------------------------------


.. py:class:: paddle.static.InputSpec(shape=None, dtype='float32', name=None)
用于描述模型输入的签名信息，包括shape、dtype和name。

此接口常用于指定高层API中模型的输入张量信息，或动态图转静态图时，指定被 ``paddle.jit.to_static`` 装饰的forward函数每个输入参数的张量信息。

参数：
  - **shape** (list|tuple)- 声明维度信息的list或tuple，默认值为None。
  - **dtype** (np.dtype|VarType|str，可选)- 数据类型，支持bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为float32。
  - **name** (str)- 被创建对象的名字，具体用法请参见 :ref:`api_guide_Name` 。

返回：初始化后的 ``InputSpec`` 对象

返回类型：InputSpec

**代码示例**

.. code-block:: python

    from paddle.static import InputSpec

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    print(input)  # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
    print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)


.. py:method:: from_tensor(tensor, name=None)
该接口将根据输入Tensor的shape、dtype等信息构建InputSpec对象。

参数：
  - **tensor** (Tensor) - 用于构建InputSpec的源Tensor
  - **name** (str): 被创建对象的名字，具体用法请参见 :ref:`api_guide_Name` 。 默认为：None。


返回：根据Tensor信息构造的 ``InputSpec`` 对象

返回类型：InputSpec


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.static import InputSpec

    paddle.disable_static()

    x = paddle.to_tensor(np.ones([2, 2], np.float32))
    x_spec = InputSpec.from_tensor(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. py:method:: from_numpy(ndarray, name=None)
该接口将根据输入numpy ndarray的shape、dtype等信息构建InputSpec对象。

参数：
  - **ndarray** (Tensor) - 用于构建InputSpec的numpy ndarray
  - **name** (str): 被创建对象的名字，具体用法请参见 :ref:`api_guide_Name` 。 默认为：None。


返回：根据ndarray信息构造的 ``InputSpec`` 对象

返回类型：InputSpec


**代码示例**

.. code-block:: python

    import numpy as np
    from paddle.static import InputSpec

    x = np.ones([2, 2], np.float32)
    x_spec = InputSpec.from_numpy(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. py:method:: batch(batch_size)
该接口将batch_size插入到当前InputSpec对象的shape元组最前面。

参数：
  - **batch_size** (int) - 被插入的batch size整型数值

返回： 更新shape信息后的 ``InputSpec`` 对象

返回类型：InputSpec


**代码示例**

.. code-block:: python

    from paddle.static import InputSpec
  
    x_spec = InputSpec(shape=[64], dtype='float32', name='x')
    x_spec.batch(4)
    print(x_spec)  # InputSpec(shape=(4, 64), dtype=VarType.FP32, name=x)


.. py:method:: unbatch()
该接口将当前InputSpec对象shape[0]值移除。


返回： 更新shape信息后的 ``InputSpec`` 对象

返回类型：InputSpec


**代码示例**

.. code-block:: python

    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
    x_spec.unbatch()
    print(x_spec)  # InputSpec(shape=(64,), dtype=VarType.FP32, name=x)

