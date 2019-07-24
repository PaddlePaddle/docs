.. _cn_api_fluid_layers_affine_grid:

affine_grid
-------------------------------

.. py:function:: paddle.fluid.layers.affine_grid(theta, out_shape, name=None)

它使用仿射变换的参数生成(x,y)坐标的网格，这些参数对应于一组点，在这些点上，输入特征映射应该被采样以生成转换后的输出特征映射。



.. code-block:: text

        * 例 1:
          给定:
              theta = [[[x_11, x_12, x_13]
                        [x_14, x_15, x_16]]
                       [[x_21, x_22, x_23]
                        [x_24, x_25, x_26]]]
              out_shape = [2, 3, 5, 5]

          Step 1:

              根据out_shape生成标准化坐标

              归一化坐标的值在-1和1之间

              归一化坐标的形状为[2,H, W]，如下所示:

              C = [[[-1.  -1.  -1.  -1.  -1. ]
                    [-0.5 -0.5 -0.5 -0.5 -0.5]
                    [ 0.   0.   0.   0.   0. ]
                    [ 0.5  0.5  0.5  0.5  0.5]
                    [ 1.   1.   1.   1.   1. ]]
                   [[-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]
                    [-1.  -0.5  0.   0.5  1. ]]]

              C[0]是高轴坐标，C[1]是宽轴坐标。

          Step2:

              将C转换并重组成形为[H * W, 2]的张量,并追加到最后一个维度

              我们得到:

              C_ = [[-1.  -1.   1. ]
                    [-0.5 -1.   1. ]
                    [ 0.  -1.   1. ]
                    [ 0.5 -1.   1. ]
                    [ 1.  -1.   1. ]
                    [-1.  -0.5  1. ]
                    [-0.5 -0.5  1. ]
                    [ 0.  -0.5  1. ]
                    [ 0.5 -0.5  1. ]
                    [ 1.  -0.5  1. ]
                    [-1.   0.   1. ]
                    [-0.5  0.   1. ]
                    [ 0.   0.   1. ]
                    [ 0.5  0.   1. ]
                    [ 1.   0.   1. ]
                    [-1.   0.5  1. ]
                    [-0.5  0.5  1. ]
                    [ 0.   0.5  1. ]
                    [ 0.5  0.5  1. ]
                    [ 1.   0.5  1. ]
                    [-1.   1.   1. ]
                    [-0.5  1.   1. ]
                    [ 0.   1.   1. ]
                    [ 0.5  1.   1. ]
                    [ 1.   1.   1. ]]
          Step3:
              按下列公式计算输出
.. math::

  Output[i] = C\_ * Theta[i]^T

参数：
  - **theta** (Variable)： 一类具有形状为[N, 2, 3]的仿射变换参数
  - **out_shape** (Variable | list | tuple)：具有格式[N, C, H, W]的目标输出的shape，out_shape可以是变量、列表或元组。
  - **name** (str|None): 此层的名称(可选)。如果没有设置，将自动命名。

返回： Variable: 形为[N, H, W, 2]的输出。

抛出异常： ValueError: 如果输入了不支持的参数类型

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    theta = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
    out_shape = fluid.layers.data(name="y", shape=[-1], dtype="float32")
    data = fluid.layers.affine_grid(theta, out_shape)
    # or
    data = fluid.layers.affine_grid(theta, [5, 3, 28, 28])









