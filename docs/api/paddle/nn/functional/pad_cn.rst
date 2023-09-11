.. _cn_api_paddle_nn_functional_pad:

pad
-------------------------------

.. py:function:: paddle.nn.functional.pad(x, pad, mode="constant", value=0.0, data_format="NCHW", name=None)

依照 ``pad`` 和 ``mode`` 属性对 ``x`` 进行 ``pad``。如果 ``mode`` 为 ``'constant'``，并且 ``pad`` 的长度为 ``x`` 维度的 2 倍时，则会根据 ``pad`` 和 ``value`` 对 ``x`` 从前面的维度向后依次补齐；否则只会对 ``x`` 在除 ``batch size`` 和 ``channel`` 之外的所有维度进行补齐。如果 ``mode`` 为 ``reflect``，则 ``x`` 对应维度上的长度必须大于对应的 ``pad`` 值。



参数
::::::::::::

  - **x** (Tensor) - Tensor，format 可以为 ``'NCL'``、``'NLC'``、``'NCHW'``、``'NHWC'``、``'NCDHW'`` 或 ``'NDHWC'``，默认值为 ``'NCHW'``，数据类型支持 float16、float32、float64、int32、int64。
  - **pad** (Tensor|list[int]|tuple[int]) - 填充大小。如果 ``mode`` 为 ``'constant'``，并且 ``pad`` 的长度为 ``x`` 维度的 2 倍时，则会根据 ``pad`` 和 ``value`` 对 ``x`` 从前面的维度向后依次补齐；否则：

     -  当输入维度为 3 时，pad 的格式为[pad_left, pad_right]；
     -  当输入维度为 4 时，pad 的格式为[pad_left, pad_right, pad_top, pad_bottom]；
     -  当输入维度为 5 时，pad 的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。

  - **mode** (str，可选) - padding 的四种模式，分别为 ``'constant'``、``'reflect'``、``'replicate'`` 和 ``'circular'``，

     - ``'constant'`` 表示填充常数 ``value``；
     - ``'reflect'`` 表示填充以 ``x`` 边界值为轴的映射；
     - ``'replicate'`` 表示填充 ``x`` 边界值；
     - ``'circular'`` 为循环填充 ``x``。具体结果可见以下示例。

  - **value** (float，可选) - 以 ``'constant'`` 模式填充区域时填充的值。默认值为 :math:`0.0`。
  - **data_format** (str，可选)  - 指定 ``x`` 的数据格式，可为 ``'NCL'``、``'NLC'``、``'NCHW'``、``'NHWC'``、``'NCDHW'`` 或 ``'NDHWC'``，默认值为 ``'NCHW'``。
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
          Out = [[[[[0., 0., 0.],
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [0., 0., 0.]]]]]

      Case 1:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'constant'
          pad_value = 0
          Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
                    [0. 0. 1. 2. 3. 0. 0.]
                    [0. 0. 4. 5. 6. 0. 0.]
                    [0. 0. 0. 0. 0. 0. 0.]]]]]

      Case 2:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'reflect'
          Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
                    [3. 2. 1. 2. 3. 2. 1.]
                    [6. 5. 4. 5. 6. 5. 4.]
                    [3. 2. 1. 2. 3. 2. 1.]]]]]

      Case 3:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'replicate'
          Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
                    [1. 1. 1. 2. 3. 3. 3.]
                    [4. 4. 4. 5. 6. 6. 6.]
                    [4. 4. 4. 5. 6. 6. 6.]]]]]

      Case 4:
          pad = [2, 2, 1, 1, 0, 0],
          mode = 'circular'
          Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
                    [2. 3. 1. 2. 3. 1. 2.]
                    [5. 6. 4. 5. 6. 4. 5.]
                    [2. 3. 1. 2. 3. 1. 2.]]]]]

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.pad
