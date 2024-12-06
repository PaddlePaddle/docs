.. _cn_api_paddle_distributed_get_mesh:

get_mesh
-------------------------------

.. py:function:: paddle.distributed.get_mesh()

获取用户通过 ``set_mesh`` 接口设定的全局 mesh。

返回
:::::::::
paddle.distributed.ProcessMesh：通过 ``set_mesh`` 接口设定的全局 mesh。


代码示例
:::::::::

COPY-FROM: paddle.distributed.get_mesh
