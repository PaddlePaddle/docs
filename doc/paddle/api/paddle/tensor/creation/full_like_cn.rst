.. _cn_api_tensor_full_like:

full_like
-------------------------------

.. py:function:: paddle.full_like(x, fill_value, dtype=None, name=None)


该OP创建一个和 ``x`` 具有相同的形状并且数据类型为 ``dtype`` 的Tensor，其中元素值均为 ``fill_value`` , 当 ``dtype`` 为None的时候，Tensor数据类型和输入 ``x`` 相同。

参数：
    - **x** (Tensor) – 输入Tensor, 输出Tensor和x具有相同的形状，x的数据类型可以是bool，float16，float32，float64，int32，int64。
    - **fill_value** (bool|float|int) - 用于初始化输出张量的常量数据的值。注意：该参数不可超过输出变量数据类型的表示范围。
    - **dtype** （np.dtype|str， 可选）- 输出变量的数据类型。若参数为None，则输出变量的数据类型和输入变量相同，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个存储结果的Tensor，数据类型和dtype相同。


    **代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    
    input = paddle.full(shape=[2, 3], fill_value=0.0, dtype='float32', name='input')
    output = paddle.full_like(input, 2.0)
    # [[2. 2. 2.]
    #  [2. 2. 2.]]

