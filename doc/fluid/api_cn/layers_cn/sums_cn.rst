.. _cn_api_fluid_layers_sums:

sums
-------------------------------

.. py:function:: paddle.fluid.layers.sums(input,out=None)

:alias_main: paddle.sums
:alias: paddle.sums,paddle.tensor.sums,paddle.tensor.math.sums
:old_api: paddle.fluid.layers.sums



该OP计算多个输入Tensor逐个元素相加的和。

- 示例：3个Tensor求和

.. code-block:: python

    输入：
        x0.shape = [2, 3]
        x0.data = [[1., 2., 3.],
                   [4., 5., 6.]]
        x1.shape = [2, 3]
        x1.data = [[10., 20., 30.],
                   [40., 50., 60.]]
        x2.shape = [2, 3]
        x2.data = [[100., 200., 300.],
                   [400., 500., 600.]]
    
    输出：
        out.shape = [2, 3]
        out.data = [[111., 222., 333.],
                    [444., 555., 666.]]

  输入：
      x0.shape = [2, 3]
      x0.data = [[1., 2., 3.],
                 [4., 5., 6.]]
      x1.shape = [2, 3]
      x1.data = [[10., 20., 30.],
                 [40., 50., 60.]]
      x2.shape = [2, 3]
      x2.data = [[100., 200., 300.],
                 [400., 500., 600.]]

  输出：
      out.shape = [2, 3]
      out.data = [[111., 222., 333.],
                  [444., 555., 666.]]


参数：
    - **input** (list) - 多个维度相同的Tensor组成的元组。支持的数据类型：float32，float64，int32，int64。
    - **out** (Variable，可选) - 指定求和的结果Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。

返回：输入的和，数据类型和维度与输入Tensor相同。若 ``out`` 为 ``None`` ，返回值是一个新的Variable；否则，返回值就是 ``out`` 。

返回类型：Variable

**代码示例**：

.. code-block:: python

    输入：
        x0.shape = [2, 3]
        x0.data = [[1., 2., 3.],
                   [4., 5., 6.]]
        x1.shape = [2, 3]
        x1.data = [[10., 20., 30.],
                   [40., 50., 60.]]
        x2.shape = [2, 3]
        x2.data = [[100., 200., 300.],
                   [400., 500., 600.]]
    
    输出：
        out.shape = [2, 3]
        out.data = [[111., 222., 333.],
                    [444., 555., 666.]]

