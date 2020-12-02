.. _cn_api_tensor_masked_select:

masked_select
-------------------------------

.. py:function:: paddle.masked_select(x, mask, name=None)



该OP返回一个1-D 的Tensor, Tensor的值是根据 ``mask`` 对输入 ``x`` 进行选择的， ``mask`` 的数据类型是bool 。

参数：
    - **x** (Tensor) - 输入Tensor, 数据类型为float32， float64，int32 或者int64。
    - **mask** (Tensor) - 用于索引的二进制掩码的Tensor，数据类型维bool。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个根据 ``mask`` 选择的的Tensor


抛出异常：
    - ``TypeError``: - 如果 ``x`` 不是 Tensor 或者 ``x`` 是Tensor的时候的数据类型不是 float32, float64, int32, int64其中之一。
    - ``TypeError``: - 如果 ``mask`` 不是 Tensor 或者 ``mask`` 是Tensor的时候的数据类型不是 bool。

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    
    data = np.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]).astype('float32')
    
    mask_data = np.array([[True, False, False, False],
                    [True, True, False, False],
                    [True, False, False, False]]).astype('bool')
    x = paddle.to_tensor(data)
    mask = paddle.to_tensor(mask_data)
    out = paddle.masked_select(x, mask)
    #[1.0 5.0 6.0 9.0]

