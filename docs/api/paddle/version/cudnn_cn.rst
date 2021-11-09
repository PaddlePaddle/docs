.. _cn_api_paddle_version_cudnn:

cudnn
-------------------------------

.. py:function:: paddle.version.cudnn()

获取 paddle 安装包编译时使用的 cuDNN 版本号。


返回：
:::::::::
若paddle wheel包为GPU版本，则返回paddle wheel包编译时使用的cuDNN的版本信息；若paddle wheel包为CPU版本，则返回 ``False`` 。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    paddle.version.cudnn()
    # '7.6.5'

