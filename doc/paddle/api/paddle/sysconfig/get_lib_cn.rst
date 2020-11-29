.. _cn_api_paddle_sysconfig_get_lib

get_lib
-------------------------------

.. py:function:: paddle.sysconfig.get_lib()

获取包含libpadle_framework的目录。

返回
::::::::::
    
    字符串类型的文件目录。

代码示例
::::::::::

.. code-block:: python

    import paddle

    include_dir = paddle.sysconfig.get_lib()
    
