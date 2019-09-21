.. _cn_api_fluid_in_dygraph_mode:

in_dygraph_mode
-------------------------------

.. py:function:: paddle.fluid.in_dygraph_mode()

检查程序是否在动态图模式中运行。

返回：如果程序是在动态图模式下运行的，则返回 ``True``。

返回类型：bool

**示例代码**

.. code-block:: python

    from __future__ import print_function
    import paddle.fluid as fluid
    if fluid.in_dygraph_mode():
        print('running in dygraph mode')
    else:
        print('not running in dygraph mode')


