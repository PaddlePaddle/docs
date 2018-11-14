.. _cn_api_fluid_layers_reduce_sum:

reduce_sum
:::::::::::::::::::::::

.. py:class:: paddle.fluid.layers.reduce_sum(input, dim=None, keep_dim=False, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

计算给定维度上张量（Tensor）元素之和。

参数：

- input（Variable）：输入变量为Tensor或LoDTensor。
- dim（list | int | None）：求和运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]
范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
- keep_dim（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
- name（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：

  减少的Tensor变量。

返回类型：

  变量（Variable）
          
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

- input（Variable）：输入变量为Tensor或LoDTensor。
- dim（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]
范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
- keep_dim（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
- name（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：

  减少的Tensor变量。

返回类型：

  变量（Variable）
          
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

- input（Variable）：输入变量为Tensor或LoDTensor。
- dim（list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求和并返回单个元素的Tensor变量，否则必须在[−rank(input),rank(input)]
范围内。如果dim [i] <0，则维度将减小为rank+dim[i]。
- keep_dim（bool | False）：是否在输出Tensor中保留减小的维度。除非keep_dim为true，否则结果张量将比输入少一个维度。
- name（str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：

  减少的Tensor变量。

返回类型：

  变量（Variable）
          
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





为输入索引序列生成一个新序列，该序列枚举输入长度为win_size的所有子序列。 枚举序列具有和可变输入第一维相同的维数，第二维是win_size，在生成中如果需要，通过设置pad_value填充。

**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]
        
参数:   

- input（Variable）: 作为索引序列的输入变量。
- win_size（int）: 枚举所有子序列的窗口大小。
- pad_value（int）: 填充值，默认为0。
          
返回:

 枚举序列变量是LoD张量（LoDTensor）。
