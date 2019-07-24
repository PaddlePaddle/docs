.. _cn_api_fluid_layers_diag:

diag
-------------------------------

.. py:function:: paddle.fluid.layers.diag(diagonal)

该功能创建一个方阵，含有diagonal指定的对角线值。

参数：
    - **diagonal** (Variable|numpy.ndarray) - 指定对角线值的输入张量，其秩应为1。

返回：存储着方阵的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        #  [3, 0, 0]
        #  [0, 4, 0]
        #  [0, 0, 5]
        import paddle.fluid as fluid
        data = fluid.layers.diag(np.arange(3, 6))




