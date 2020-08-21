.. _cn_api_nn_cn_pad:

pad
-------------------------------

.. py:function:: paddle.nn.functional.pad(x, pad, mode="constant", value=0.0, data_format="NCHW", name=None)

该OP依照 pad 和 mode 属性对input进行 ``pad`` 。

参数：
  - **x** (Tensor) - Tensor，format可以为 `'NCL'`, `'NLC'`, `'NCHW'`, `'NHWC'`, `'NCDHW'`
    或 `'NDHWC'`，默认值为`'NCHW'`，数据类型支持float16, float32, float64, int32, int64。
  - **pad** (Tensor | List[int32]) - 填充大小。当输入维度为3时，pad的格式为[pad_left, pad_right]；
    当输入维度为4时，pad的格式为[pad_left, pad_right, pad_top, pad_bottom]；
    当输入维度为5时，pad的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
  - **mode** (str) - padding的四种模式，分别为 `'constant'`, `'reflect'`, `'replicate'` 和`'circular'`。
    `'constant'` 表示填充常数 `value`；`'reflect'` 表示填充以input边界值为轴的映射；`'replicate'` 表示
    填充input边界值；`'circular'`为循环填充input。具体结果可见以下示例。默认值为 `'constant'` 。
  - **value** (float32) - 以 `'constant'` 模式填充区域时填充的值。默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 `'NCL'`, `'NLC'`, `'NCHW'`, `'NHWC'`, `'NCDHW'`
    或 `'NDHWC'`，默认值为`'NCHW'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。
返回： 对input进行``pad`` 的结果，数据类型和input相同。

返回类型：Tensor

**示例**：

.. code-block:: text

      x = [[[[[1., 2., 3.],
              [4., 5., 6.]]]]]

      Case 0:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'constant'
          pad_value = 0
          Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
                    [0. 0. 1. 2. 3. 0. 0.]
                    [0. 0. 4. 5. 6. 0. 0.]
                    [0. 0. 0. 0. 0. 0. 0.]]]]]

      Case 1:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'reflect'
          Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
                    [3. 2. 1. 2. 3. 2. 1.]
                    [6. 5. 4. 5. 6. 5. 4.]
                    [3. 2. 1. 2. 3. 2. 1.]]]]]

      Case 2:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'replicate'
          Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
                    [1. 1. 1. 2. 3. 3. 3.]
                    [4. 4. 4. 5. 6. 6. 6.]
                    [4. 4. 4. 5. 6. 6. 6.]]]]]

      Case 3:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'circular'
          Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
                    [2. 3. 1. 2. 3. 1. 2.]
                    [5. 6. 4. 5. 6. 4. 5.]
                    [2. 3. 1. 2. 3. 1. 2.]]]]]

**代码示例：**

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn.functional as F

    paddle.disable_static()

    # example 1
    x_shape = (1, 1, 3)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
    tensor_x = paddle.to_tensor(x)
    y = F.pad(tensor_x, [2, 3], value=1, mode='constant')
    print(y.numpy())
    # [[[1. 1. 1. 2. 3. 1. 1. 1.]]]

    # example 2
    x_shape = (1, 1, 2, 3)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
    tensor_x = paddle.to_tensor(x)
    y = F.pad(tensor_x, [1, 2, 1, 1], value=1, mode='circular')
    print(y.numpy())
    # [[[[6. 4. 5. 6. 4. 5.]
    #    [3. 1. 2. 3. 1. 2.]
    #    [6. 4. 5. 6. 4. 5.]
    #    [3. 1. 2. 3. 1. 2.]]]]



