.. _cn_api_fluid_layers_reduce_sum:

reduce_sum
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_sum(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素之和。

参数：
          - **input**（Variable）：输入变量为Tensor或LoDTensor。
          - **dim**（list | int | None）：求和运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
          - **keep_dim**（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
          - **name**（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x is a Tensor variable with following elements:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # Each example is followed by the corresponding output tensor.
      fluid.layers.reduce_sum(x)  # [3.5]
      fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
      fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
      fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

      # x is a Tensor variable with shape [2, 2, 2] and elements as below:
      #      [[[1, 2], [3, 4]],
      #      [[5, 6], [7, 8]]]
      # Each example is followed by the corresponding output tensor.
      fluid.layers.reduce_sum(x, dim=[1, 2]) # [10, 26]
      fluid.layers.reduce_sum(x, dim=[0, 1]) # [16, 20]
      

.. _cn_api_fluid_layers_reduce_prod:

reduce_prod
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素乘积。

参数：
          - **input**（Variable）：输入变量为Tensor或LoDTensor。
          - **dim**（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
          - **keep_dim**（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
          - **name**（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x is a Tensor variable with following elements:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_prod(x)  # [0.0002268]
      fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
      fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
      fluid.layers.reduce_prod(x, dim=1,
                               keep_dim=True)  # [[0.027], [0.0084]]

      # x is a Tensor variable with shape [2, 2, 2] and elements as below:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_prod(x, dim=[1, 2]) # [24.0, 1680.0]
      fluid.layers.reduce_prod(x, dim=[0, 1]) # [105.0, 384.0]


.. _cn_api_fluid_layers_reduce_min:

reduce_min
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素乘积。

参数：
          - **input**（Variable）：输入变量为Tensor或LoDTensor。
          - **dim**（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
          - **keep_dim**（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
          - **name**（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x is a Tensor variable with following elements:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_min(x)  # [0.1]
      fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
      fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
      fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

      # x is a Tensor variable with shape [2, 2, 2] and elements as below:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_min(x, dim=[1, 2]) # [1.0, 5.0]
      fluid.layers.reduce_min(x, dim=[0, 1]) # [1.0, 2.0]


.. _cn_api_fluid_layers_reduce_mean:

reduce_mean
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_mean(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素平均值。

参数：
          - **input**（Variable）：输入变量为Tensor或LoDTensor。
          - **dim**（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
          - **keep_dim**（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
          - **name**（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x is a Tensor variable with following elements:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_mean(x)  # [0.4375]
      fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
      fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
      fluid.layers.reduce_mean(
          x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

      # x is a Tensor variable with shape [2, 2, 2] and elements as below:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_mean(x, dim=[1, 2]) # [2.5, 6.5]
      fluid.layers.reduce_mean(x, dim=[0, 1]) # [4.0, 5.0]


.. _cn_api_fluid_layers_reduce_max:

reduce_max
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_max(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素最大值。

参数：
          - **input**（Variable）：输入变量为Tensor或LoDTensor。
          - **dim**（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
          - **keep_dim**（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
          - **name**（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少的Tensor变量。

返回类型：  变量（Variable）
          
**代码示例**

..  code-block:: python

      # x is a Tensor variable with following elements:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_max(x)  # [0.9]
      fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
      fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
      fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

      # x is a Tensor variable with shape [2, 2, 2] and elements as below:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # Each example is followed by the correspending output tensor.
      fluid.layers.reduce_max(x, dim=[1, 2]) # [4.0, 8.0]
      fluid.layers.reduce_max(x, dim=[0, 1]) # [7.0, 8.0]


.. _cn_api_fluid_layers_prelu:

prelu
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.prelu(x, mode, param_attr=None, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

等式：

.. math::
    $$y = max(0, x) + alpha min(0, x)$$

参数：
          - **x**（Variable）：输入为Tensor。
          - **param_attr**(ParamAttr|None)：可学习权重 :math:`\[\alpha\]`参数属性。
          - **mode**（string）:权重共享的模式all：所有元素共享相同的权重通道：通道中的元素共享相同的权重元素：每个元素都有一个权重
          - **name**（str | None）:这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回： 输出Tensor与输入shape相同。

返回类型：  变量（Variable）
  
  
  
.. _cn_api_fluid_layers_pad_constant_like:

pad_constant_like
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.pad_constant_like(x, y, pad_value=0.0, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

使用pad_value填充输入（Y），填充到每个axis边缘值的数量由X和Y的shape不同而指定。（（0，shape_x_0 - shape_y_0），...（0，shape_x_n - shape_y_n ））是每个axis唯一pad宽度。输入应该是k-D张量（k> 0且k <7）。

**实例如下**

:

    Given:
        X = [[[[ 0,  1,  2],
               [ 3,  4,  5]],
              [[ 6,  7,  8],
               [ 9, 10, 11]],
              [[12, 13, 14],
               [15, 16, 17]]],
             [[[18, 19, 20],
               [21, 22, 23]],
              [[24, 25, 26],
               [27, 28, 29]],
              [[30, 31, 32],
               [33, 34, 35]]]]
        X.shape = (2, 3, 2, 3)

        Y = [[[[35, 36, 37]],
              [[38, 39, 40]],
              [[41, 42, 43]]]]
        Y.shape = (1, 3, 1, 3)
        
参数：
          - **x**（Variable）：输入Tensor变量。
          - **y**（Variable）：输出Tensor变量。
          - **pad_value** (float)：用于填充的常量值。
          - **name**（str | None）:这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：填充张量（Tensor）变量。。

返回类型：  变量（Variable）

**示例代码**

..  code-block:: python

    # x is a rank 4 tensor variable, x.shape = (2, 3, 2, 3)
    # y is a rank 4 tensor variable, y.shape = (1, 3, 1, 3)
    out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
    # out is a rank 4 tensor variable, and out.shape = [2, 3 ,2 , 3]






.. math::
    $$y = max(0, x) + alpha min(0, x)$$

参数：
          - **x**（Variable）：输入为Tensor。
          - **param_attr**(ParamAttr|None)：可学习权重 :math:`\[\alpha\]`参数属性。
          - **mode**（string）:权重共享的模式all：所有元素共享相同的权重通道：通道中的元素共享相同的权重元素：每个元素都有一个权重
          - **name**（str | None）:这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回： 输出Tensor与输入shape相同。

返回类型：  变量（Variable）
  
  
  
  
