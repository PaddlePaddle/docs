.. _cn_api_tensor_full_like:

full_like
-------------------------------

.. py:function:: paddle.full_like(input, fill_value, out=None, dtype=None, device=None, stop_gradient=True, name=None)

该OP创建一个和input具有相同的形状和数据类型的Tensor，其中元素值均为fill_value。

参数：
    - **input** (Variable) – 指定输入为一个多维的Tensor，数据类型可以是bool，float16，float32，float64，int32，int64。
    - **fill_value** (bool|float|int) - 用于初始化输出Tensor的常量数据的值。默认为0。注意：该参数不可超过输出变量数据类型的表示范围。
    - **out** (Variable，可选) - 输出Tensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。
    - **dtype** （np.dtype|core.VarDesc.VarType|str， 可选）- 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为None。
    - **device** (str，可选) – 选择在哪个设备运行该操作，可选值包括None，'cpu'和'gpu'。如果 ``device`` 为None，则将选择运行Paddle程序的设备，默认为None。
    - **stop_gradient** (bool，可选) – 是否从此 Variable 开始，之前的相关部分都停止梯度计算，默认为True。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个存储结果的Tensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    input = fluid.data(name='input', dtype='float32', shape=[2, 3])
    output = paddle.full_like(input, 2.0)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    img=np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    res = exe.run(fluid.default_main_program(), feed={'input':img}, fetch_list=[output])
    print(res) # [array([[2., 2., 2.], [2., 2., 2.]], dtype=float32)]

