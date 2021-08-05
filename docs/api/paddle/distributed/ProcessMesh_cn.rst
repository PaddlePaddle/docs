.. _cn_api_distributed_ProcessMesh:

ProcessMesh
-------------------------------


.. py:class:: paddle.distributed.ProcessMesh(mesh, parent=None)



mesh是表示逻辑进程组织结构的N-维数组。该数组的形状表示逻辑进程的拓扑结构,
数组的元素值表示某个逻辑进程。例如，下面的图例可以用N-维数组[[2, 4, 5], [0, 1, 3]]
表示，第一个逻辑进程的id是2。

| 2 | 4 | 5 |

| 0 | 1 | 3 |

参数：
    - **mesh** (numpy.ndarray) - 表示进程的N-维数组，值类型为int。
    - **parent** (ProcessMesh, 可选) - 父ProcessMesh，None表示没有父ProcessMesh。默认值为None。

返回值：
    None。

抛出异常：
    ValueError: 如果mesh不是numpy.ndarray实例。

**代码示例**:

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.distributed as dist
    
    paddle.enable_static()
    
    mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
    assert mesh.parent is None
    assert mesh.topology == [2, 3]
    assert mesh.process_group == [2, 4, 5, 0, 1, 3]
    mesh.set_placement([0, 1, 2, 3, 4, 5])

   

属性
::::::::::::
.. py:attribute:: topology
ProcessMesh表示的进程拓扑结构，类型为list。

.. py:attribute:: process_group
ProcessMesh表示的所有进程，类型为list。

.. py:attribute:: parent
ProcessMesh的父ProcessMesh，类型为ProcessMesh。

方法
::::::::::::
.. py:method:: set_placement(order)
设置物理进程的顺序。

参数：
    - **order** (list): 物理进程id的顺序

返回：
   None。


**代码示例**:

.. code-block:: python

   import numpy as np
   import paddle
   import paddle.distributed as dist

   paddle.enable_static()

   mesh = dist.ProcessMesh(np.array([[2, 4, 5], [0, 1, 3]]))
   mesh.set_placement([0, 1, 2, 3, 4, 5])
