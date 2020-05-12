.. _cn_api_tensor_random_randperm:

randperm
-------------------------------

.. py:function:: paddle.tensor.random.randperm(n, out=None, dtype="int64", device=None, stop_gradient=True, seed=0)

该OP返回一个数值在0到n-1、顺序随机的整数排列。

参数: 
  - **n** (int): 整数排列的上限，应该大于0。 
  - **out** (Variable, optional): 可选的输出变量，如果不为 `None` ，返回的整数排列保存在该变量中，默认是 `None` 。
  - **dtype** (np.dtype|core.VarDesc.VarType|str, optional): 整数排列的数据类型，支持 `int64` 和 `int32` ，默认是 `int64` 。
  - **device** (str, optional): 指定整数排列所在的设备内存。设置为 `cpu` 则保存在 `cpu` 内存中，设置为 `gpu` ，则保存在 `gpu` 内存中，设置为 `None` 则保存在运行的设备内存中。默认是 `None` 。
  - **stop_gradient** (bool, optional): 返回的整数排列是否记录并更新梯度，默认是 `True` 。 
  - **seed** (int, optional): 设置随机种子。`seed` 等于0时，每次返回不同的整数排列；`seed` 不等于0时，相同的 `seed` 返回相同的整数排列。

返回:  一个数值在0到n-1、顺序随机的整数排列。

返回类型: Variable

**代码示例**:

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    # Note that, the random permutation returned by randperm depends
    # the random seed in computer, so the output in the next example
    # will be change.
    with fluid.dygraph.guard():
        out_1 = paddle.randperm(6)
        print(out_1.numpy())  # Random permutation, for example [2 4 5 0 3 1]

        out_2 = fluid.dygraph.to_variable(
				np.array([0, 1, 2, 3])).astype(np.int64)
        paddle.randperm(6, out_2)
        print(out_2.numpy())  # Random permutation, for example [5 0 2 4 1 3]

        out_3 = paddle.randperm(6, dtype="int32", device="cpu")
        print(out_3.numpy())  # Random permutation, for example [3 1 4 2 5 0]

        out_4 = paddle.randperm(6, device="cpu", stop_gradient=True)
        print(out_4.numpy())  # Random permutation, for example [3 1 5 2 0 4]     
