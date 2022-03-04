.. _cn_api_nn_Mish:

Mish
-------------------------------
.. py:class:: paddle.nn.Mish(name=None)

Mish激活层

.. math::

        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        Mish(x) = x * \tanh(softplus(x))


参数
::::::::::
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-5., 0., 5.])
    m = paddle.nn.Mish()
    out = m(x) # [-0.03357624, 0., 4.99955208]
