.. _cn_api_paddle_nn_functional_pad:

pad
-------------------------------

.. py:function:: paddle.nn.functional.pad(x, pad, mode="constant", value=0.0, data_format="NCHW", pad_from_left_axis=True, name=None)

依照 ``pad`` 和 ``mode`` 属性对 ``x`` 进行 ``pad``。

.. note::
    1. 记 ``x`` 的维数为 N (以下延用)。当 ``mode`` 为 ``'constant'`` 时， ``pad`` 的长度可以是任意小于等于 2*N 的偶数。
    2. pad 的顺序支持右对齐(从 ``x`` 的最后一维开始)。特别地，当 ``mode`` 为 ``'constant'`` ，且 ``pad`` 是长度为 2N 的列表时，pad 的顺序可以通过 ``pad_from_left_axis`` 参数来控制，如果 ``pad_from_left_axis`` 是 True，pad 的顺序则是左对齐；如果 ``pad_from_left_axis`` 是 False，pad 的顺序则是右对齐。
    3. 当 ``mode`` 为 ``'reflect'``、 ``'replicate'``、 ``'circular'``，或 ``pad`` 是 Tensor，或 ``pad`` 的长度是 2*(N-2) 时，``x`` 的维数只支持 3-D、4—D、5-D。此时 pad 作用在相应 ``data_format`` 的 [D, H, W] 轴上，顺序是从 [D, H, W] 轴的最后一维到第一维。具体地，当 N=3 时，pad 的格式为[pad_left, pad_right]；当 N=4 时，pad 的格式为[pad_left, pad_right, pad_top, pad_bottom]；当 N=5 时，pad 的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
    4. 如果 ``mode`` 为 ``reflect``，则 ``x`` 对应 [D, H, W] 维度上的长度必须大于对应的 ``pad`` 值。

参数
::::::::::::

  - **x** (Tensor) - Tensor，format 可以为 ``'NCL'``、``'NLC'``、``'NCHW'``、``'NHWC'``、``'NCDHW'`` 或 ``'NDHWC'``，默认值为 ``'NCHW'``，数据类型支持 float16、float32、float64、int32、int64、complex64、complex128。
  - **pad** (Tensor|list[int]|tuple[int]) - 填充大小，基本数据类型是整数类型。具体设置请参照 Note（注解）。

  - **mode** (str，可选) - padding 的四种模式，分别为 ``'constant'``、``'reflect'``、``'replicate'`` 和 ``'circular'``，

     - ``'constant'`` 表示填充常数 ``value``；
     - ``'reflect'`` 表示填充以 ``x`` 边界值为轴的映射；
     - ``'replicate'`` 表示填充 ``x`` 边界值；
     - ``'circular'`` 为循环填充 ``x``。具体结果可见以下示例。

  - **value** (float，可选) - 以 ``'constant'`` 模式填充区域时填充的值。默认值为 :math:`0.0`。
  - **data_format** (str，可选) - 当 ``mode`` 为 ``'reflect'``、 ``'replicate'``、 ``'circular'``，或 ``pad`` 是 Tensor，或 ``pad`` 的长度是 2*(N-2) 时，指定 ``x`` 的数据格式，可为 ``'NCL'``、``'NLC'``、``'NCHW'``、``'NHWC'``、``'NCDHW'`` 或 ``'NDHWC'``，默认值为 ``'NCHW'``。
  - **pad_from_left_axis** (bool，可选) - 只有当 ``mode`` 为 ``'constant'`` ，且 ``pad`` 是长度为 2N 的列表时有效，设置 ``pad`` 与 ``x`` 的轴左对齐或右对齐。默认值为 True，表示左对齐填充。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
返回
::::::::::::
Tensor，对 ``x`` 进行 ``'pad'`` 的结果，数据类型和 ``x`` 相同。


**示例**：

.. code-block:: text

      x = [[[[[1., 2., 3.],
              [4., 5., 6.]]]]]

      Case 0:
          pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          mode = 'constant'
          value = 0
          pad_from_left_axis = True
          Out = [[[[[0., 0., 0.],
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [0., 0., 0.]]]]]
          Out.shape = [1, 1, 1, 4, 3]

      Case 1:
          pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          mode = 'constant'
          value = 0
          pad_from_left_axis = False
          Out = [[[[[0., 0., 0.],
                      [0., 0., 0.]]],
                  [[[1., 2., 3.],
                      [4., 5., 6.]]],
                  [[[0., 0., 0.],
                      [0., 0., 0.]]]]]
          Out.shape = [1, 3, 1, 2, 3]

      Case 2:
          pad = [1, 0, 0, 1],
          mode = 'constant'
          value = 0
          Out = [[[[[0., 1., 2., 3.],
                      [0., 4., 5., 6.],
                      [0., 0., 0., 0.]]]]]
          Out.shape = [1, 1, 1, 3, 4]

      Case 3:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'constant'
          value = 0
          Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
                      [0. 0. 1. 2. 3. 0. 0.]
                      [0. 0. 4. 5. 6. 0. 0.]
                      [0. 0. 0. 0. 0. 0. 0.]]]]]
          Out.shape = [1, 1, 1, 4, 7]

      Case 4:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'reflect'
          Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
                      [3. 2. 1. 2. 3. 2. 1.]
                      [6. 5. 4. 5. 6. 5. 4.]
                      [3. 2. 1. 2. 3. 2. 1.]]]]]
          Out.shape = [1, 1, 1, 4, 7]

      Case 5:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'replicate'
          Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
                      [1. 1. 1. 2. 3. 3. 3.]
                      [4. 4. 4. 5. 6. 6. 6.]
                      [4. 4. 4. 5. 6. 6. 6.]]]]]
          Out.shape = [1, 1, 1, 4, 7]

      Case 6:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'circular'
          Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
                      [2. 3. 1. 2. 3. 1. 2.]
                      [5. 6. 4. 5. 6. 4. 5.]
                      [2. 3. 1. 2. 3. 1. 2.]]]]]
          Out.shape = [1, 1, 1, 4, 7]

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.pad
