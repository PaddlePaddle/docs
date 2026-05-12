.. _cn_api_paddle__assert:

_assert
-------------------------------

.. py:function:: paddle._assert(condition, message="")

对 Python assert 的封装，支持符号追踪。

在动态图模式下，该函数的行为与 Python 的 assert 语句一致。在静态图模式下，当 condition 是 Tensor 时，会在计算图中创建一个 Assert 算子。

参数
::::::::::::
    - **condition** (bool|Tensor) - 断言的条件。如果是 Tensor，必须是布尔标量（numel=1）。
    - **message** (str，可选) - 断言失败时显示的错误信息。默认值为 ""。

代码示例
::::::::::::

.. code-block:: pycon

    >>> import paddle
    >>> # Non-tensor condition
    >>> paddle._assert(1 == 1, "This should pass")

    >>> # Tensor condition
    >>> x = paddle.to_tensor([True])
    >>> paddle._assert(x, "Tensor assertion")
