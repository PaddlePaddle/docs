.. _cn_api_paddle_compat_to_bytes

to_bytes
-------------------------------

.. py:function:: paddle.compat.to_bytes(obj, encoding='utf-8', inplace=False)

飞桨中的所有字符串都需要用文本字符串表示。
此函数将对象转换为具有特定编码的字节。特别是，如果对象类型是列表或集合容器，我们将迭代对象中的所有项并将其转换为字节。

在Python3中:
    使用特定编码将str type对象编码为bytes类型。

在Python2中:
    使用特定的编码将unicode类型的对象编码为str类型，或者只返回object的8位字符串。

参数
::::::::::
    
    - **obj** (unicode|str|bytes|list|set) - 要编码的对象。
    - **encoding** (str) - 对字符串进行编码的编码格式。
    - **inplace** (bool) - 是否改变原始对象或创建一个新对象。

返回
::::::::::
    
    obj解码后的结果。

代码示例
:::::::::

.. code-block:: python

    import paddle

    data = "paddlepaddle"
    data = paddle.compat.to_bytes(data)
    # b'paddlepaddle'
