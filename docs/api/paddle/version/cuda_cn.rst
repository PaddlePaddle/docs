.. _cn_api_paddle_version_cuda:

cuda
-------------------------------

.. py:function:: paddle.version.cuda()

获取 paddle 安装包使用的 cuda 版本号。


返回：
:::::::::
返回cuda的版本信息。若paddle wheel包为GPU版本，则返回cuda具体的版本号；若paddle wheel包为CPU版本，则返回 ``False`` 。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    paddle.version.cuda()
    # '10.2'

