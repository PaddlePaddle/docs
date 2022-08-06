.. _cn_api_paddle_compat_to_bytes:

to_bytes
-------------------------------

.. py:function:: paddle.compat.to_bytes(obj, encoding='utf-8', inplace=False)

飞桨中的所有字符串都需要用文本字符串表示。
此函数将对象转换为具有特定编码的字节。特别是，如果对象类型是列表或集合容器，我们将迭代对象中的所有项并将其转换为字节。

在 Python3 中：

    使用特定编码将 str type 对象编码为 bytes 类型。

在 Python2 中：

    使用特定的编码将 unicode 类型的对象编码为 str 类型，或者只返回 object 的 8 位字符串。

参数
::::::::::

    - **obj** (unicode|str|bytes|list|set) - 要编码的对象。
    - **encoding** (str) - 对字符串进行编码的编码格式。
    - **inplace** (bool) - 是否改变原始对象或创建一个新对象。

返回
::::::::::

    obj 解码后的结果。

代码示例
:::::::::

COPY-FROM: paddle.compat.to_bytes
