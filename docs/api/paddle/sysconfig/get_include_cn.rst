.. _cn_api_paddle_sysconfig_get_include

get_include
-------------------------------

.. py:function:: paddle.sysconfig.get_include()

获取包含飞桨C++头文件的目录。

返回
::::::::::
    
    字符串类型的文件目录。

代码示例
::::::::::

.. code-block:: python

    import paddle

    include_dir = paddle.sysconfig.get_include()
    
