.. _cn_api_paddle_static_name_scope:

name_scope
-------------------------------


.. py:function:: paddle.static.name_scope(prefix=None)

该函数为静态图下的 operators 生成不同的命名空间。

.. note::
    该函数只用于静态图下的调试和可视化，不建议用在其它方面，否则会引起内存泄露。


参数
::::::::::::

  - **prefix** (str，可选) - 名称前缀。默认值为 None。

代码示例
::::::::::::

COPY-FROM: paddle.static.name_scope
