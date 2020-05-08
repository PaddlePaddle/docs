.. _cn_api_tensor_norm:

norm
-------------------------------

.. py:function:: paddle.norm(input, p='fro', axis=None, keepdim=False, out=None, name=None):

该OP将计算给定Tensor的矩阵范数（Frobenius 范数）和向量范数（向量1范数、2范数、或者通常的p范数）.

参数：
    - **input** (Variable) - 输入Tensor。维度为多维，数据类型为float32或float64。
    - **p** (float|string, 可选) - 范数的种类。目前支持的值为 `fro`、 `1`、 `2`，和任何正实数p对应的p范数。
    - **axis** (int|list, 可选) - 使用范数计算的轴。如果 ``axis`` 为int或者只有一个元素的list，``norm`` API会计算输入Tensor的向量范数。如果axis为包含两个元素的list，API会计算输入Tensor的矩阵范数。 当 ``axis < 0`` 时，实际的计算维度为 rank(input) + axis。
    - **keepdim** (bool，可选) - 是否在输出的Tensor中保留和输入一样的维度，默认值为False。当 :attr:`keepdim` 为False时，输出的Tensor会比输入 :attr:`input` 的维度少一些。 
    - **out** (Variable，可选) - 指定输出的Tensor，默认值为None。out的数据类型必须与输入 ``input`` 一致。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

    返回：在指定axis上进行范数计算的Tensor，与输入input数据类型相同。

    返回类型：Variable，与输入input数据类型相同。

抛出异常：
    - ``TypeError`` - 当输出 ``out`` 和输入 ``input`` 数据类型不一致时候。
    - ``ValueError`` - 当参数  ``p`` 或者 ``axis`` 不合法时。

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    x = fluid.data(name='x', shape=[2, 3, 5], dtype='float64')
    
    # compute frobenius norm along last two dimensions.
    out_fro = paddle.norm(x, p='fro', axis=[1,2])
    
    # compute 2-order vector norm along last dimension.
    out_pnorm = paddle.norm(x, p=2, axis=-1)
