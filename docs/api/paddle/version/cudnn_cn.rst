.. _cn_api_paddle_version_cudnn:

cudnn
-------------------------------

.. py:function:: paddle.version.cudnn()

获取 paddle 安装包使用的 cudnn 版本号。


返回：
:::::::::
返回cudnn的版本信息。若paddle包为CPU版本，则返回``False``。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    paddle.version.cudnn()
    # '7.6.5'

