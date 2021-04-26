.. _cn_api_tensor_norm:

norm
-------------------------------

.. py:function:: paddle.norm(x, p='fro', axis=None, keepdim=False, name=None):




该OP将计算给定Tensor的矩阵范数（Frobenius 范数）和向量范数（向量1范数、2范数、或者通常的p范数）.

.. note::

    此API与`numpy.linalg.norm`存在差异。此API支持高阶张量（rank>=3）作为输入，输入`axis`对应的轴就可以计算出norm的值。但是`numpy.linalg.norm`仅支持一维向量和二维矩阵作为输入。特别需要注意的是，此API的P阶矩阵范数，实际上将矩阵摊平成向量计算。实际计算的是向量范数，而不是真正的矩阵范数。

参数
:::::::::

    - **x** (Tensor) - 输入Tensor。维度为多维，数据类型为float32或float64。
    - **p** (float|string, 可选) - 范数(ord)的种类。目前支持的值为 `fro`、`inf`、`-inf`、`0`、`1`、`2`，和任何正实数p对应的p范数。默认值为 `fro` 。
    - **axis** (int|list|tuple, 可选) - 使用范数计算的轴。如果 ``axis`` 为None，则忽略input的维度，将其当做向量来计算。如果 ``axis`` 为int或者只有一个元素的list|tuple，``norm`` API会计算输入Tensor的向量范数。如果axis为包含两个元素的list，API会计算输入Tensor的矩阵范数。 当 ``axis < 0`` 时，实际的计算维度为 rank(input) + axis。默认值为 `None` 。
    - **keepdim** (bool，可选) - 是否在输出的Tensor中保留和输入一样的维度，默认值为False。当 :attr:`keepdim` 为False时，输出的Tensor会比输入 :attr:`input` 的维度少一些。 
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

返回
:::::::::

    - 在指定axis上进行范数计算的Tensor，与输入input数据类型相同。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np
    shape=[2, 3, 4]
    np_input = np.arange(24).astype('float32') - 12
    np_input = np_input.reshape(shape)
    x = paddle.to_tensor(np_input)
    #[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
    # [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]

    # compute frobenius norm along last two dimensions.
    out_fro = paddle.norm(x, p='fro', axis=[0,1])
    # out_fro.numpy() [17.435596 16.911535 16.7332   16.911535]

    # compute 2-order vector norm along last dimension.
    out_pnorm = paddle.norm(x, p=2, axis=-1)
    #out_pnorm.numpy(): [[21.118711  13.190906   5.477226]
    #                    [ 3.7416575 11.224972  19.131126]]

    # compute 2-order  norm along [0,1] dimension.
    out_pnorm = paddle.norm(x, p=2, axis=[0,1])
    #out_pnorm.numpy(): [17.435596 16.911535 16.7332   16.911535]

    # compute inf-order  norm
    out_pnorm = paddle.norm(x, p=np.inf)
    #out_pnorm.numpy()  = [12.]
    out_pnorm = paddle.norm(x, p=np.inf, axis=0)
    #out_pnorm.numpy(): [[12. 11. 10. 9.] [8. 7. 6. 7.] [8. 9. 10. 11.]]

    # compute -inf-order  norm
    out_pnorm = paddle.norm(x, p=-np.inf)
    #out_pnorm.numpy(): [0.]
    out_pnorm = paddle.norm(x, p=-np.inf, axis=0)
    #out_pnorm.numpy(): [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]
 
