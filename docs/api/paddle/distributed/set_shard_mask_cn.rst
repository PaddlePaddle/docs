.. _cn_api_distributed_set_shard_mask:

set_shard_mask
-------------------------------


.. py:function:: paddle.distributed.set_shard_mask(x, mask)

设置输入Tensor `x` 的掩码信息，表示输入`x`在其ProcessMesh实例表示的逻辑进程中是否存在。

参数
:::::::::
    - x (Tensor) - 待处理的输入Tensor。
    - mask (numpy.ndarray) - 其形状必须和输入`x`的ProcessMesh实例相同，其元素值必须是0或者1。这里，值1表示输入`x`在对应的逻辑进程中存在，值0表示输入`x`在对应的逻辑进程中不存在。例如，下面的示例N-维数组[[2, 4, 5], [0, 1, 3]]。

       | 2 | 4 | 5 |

       | 0 | 1 | 3 |

       并且，下面的例子表示给定的`mask`。
       
       | 1 | 0 | 1 |

       | 0 | 1 | 0 |

       那么，输入`x`存在于逻辑进程2、5和1中，而不存在于逻辑进程4、0和3中。

返回
:::::::::
Tensor: 输入`x`自身。

代码示例
:::::::::
.. code-block:: python

        import numpy as np
        import paddle
        import paddle.distributed as dist

        paddle.enable_static()

        mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
        x = paddle.ones([4, 6])
        dist.set_shard_mask(x, mask)
