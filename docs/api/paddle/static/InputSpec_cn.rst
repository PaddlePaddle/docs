.. _cn_api_static_cn_InputSpec:

InputSpec
-------------------------------


.. py:class:: paddle.static.InputSpec(shape=None, dtype='float32', name=None)
用于描述模型输入的签名信息，包括 shape、dtype 和 name。

此接口常用于指定高层 API 中模型的输入 Tensor 信息，或动态图转静态图时，指定被 ``paddle.jit.to_static`` 装饰的 forward 函数每个输入参数的 Tensor 信息。

参数
::::::::::::

  - **shape** (list|tuple)- 声明维度信息的 list 或 tuple，默认值为 None。
  - **dtype** (np.dtype|str，可选)- 数据类型，支持 bool，float16，float32，float64，int8，int16，int32，int64，uint8。默认值为 float32。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
初始化后的 ``InputSpec`` 对象。


代码示例
::::::::::::

.. code-block:: python

    from paddle.static import InputSpec

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    print(input)  # InputSpec(shape=(-1, 784), dtype=paddle.float32, name=x)
    print(label)  # InputSpec(shape=(-1, 1), dtype=paddle.int64, name=label)


方法
::::::::::::
from_tensor(tensor, name=None)
'''''''''
该接口将根据输入 Tensor 的 shape、dtype 等信息构建 InputSpec 对象。

**参数**

  - **tensor** (Tensor) - 用于构建 InputSpec 的源 Tensor
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


**返回**

根据 Tensor 信息构造的 ``InputSpec`` 对象。


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.static import InputSpec

    x = paddle.to_tensor(np.ones([2, 2], np.float32))
    x_spec = InputSpec.from_tensor(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)


from_numpy(ndarray, name=None)
'''''''''
该接口将根据输入 numpy ndarray 的 shape、dtype 等信息构建 InputSpec 对象。

**参数**

  - **ndarray** (Tensor) - 用于构建 InputSpec 的 numpy ndarray
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


**返回**

根据 ndarray 信息构造的 ``InputSpec`` 对象。


**代码示例**

.. code-block:: python

    import numpy as np
    from paddle.static import InputSpec

    x = np.ones([2, 2], np.float32)
    x_spec = InputSpec.from_numpy(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)


batch(batch_size)
'''''''''
该接口将 batch_size 插入到当前 InputSpec 对象的 shape 元组最前面。

**参数**

  - **batch_size** (int) - 被插入的 batch size 整型数值

**返回**

 更新 shape 信息后的 ``InputSpec`` 对象。


**代码示例**

.. code-block:: python

    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[64], dtype='float32', name='x')
    x_spec.batch(4)
    print(x_spec)  # InputSpec(shape=(4, 64), dtype=paddle.float32, name=x)


unbatch()
'''''''''
该接口将当前 InputSpec 对象 shape[0]值移除。


**返回**

 更新 shape 信息后的 ``InputSpec`` 对象。


**代码示例**

.. code-block:: python

    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
    x_spec.unbatch()
    print(x_spec)  # InputSpec(shape=(64,), dtype=paddle.float32, name=x)
