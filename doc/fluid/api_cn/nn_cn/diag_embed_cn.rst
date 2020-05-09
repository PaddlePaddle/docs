.. _cn_api_fluid_layers_nn_diag_embed:

diag_embed
-------------------------------

.. py:function:: fluid.layers.diag_embed(input, offset=0, dim1=-2, dim2=-1):

    该 OP 创建一个 Tensor，其在指定的 2D 平面（由 ``dim1`` 和 ``dim2`` 指定）上的对角线由输入 ``input`` 填充。
    默认的，指定的 2D 平面由返回 Tensor 的最后两维组成。
    
    参数 ``offset`` 确定在指定的二维平面中填充对角线的位置：

    - 如果 offset = 0，则填充主对角线。
    - 如果 offset > 0，则填充主对角线右上的对角线。
    - 如果 offset < 0，则填充主对角线左下的对角线。

参数：
    - **input** （Variable|numpy.ndarray）- 输入变量，至少为 1D 数组，支持数据类型为 float32，float64，int32，int64。
    - **offset** （int ，可选）- 从指定的二维平面中获取对角线的位置，默认值为 0，既主对角线。
    - **dim1** （int ， 可选）- 填充对角线的二维平面的第一维，默认值为 -2。
    - **dim2** （int ， 可选）- 填充对角线的二维平面的第二维，默认值为 -1。

返回： 指定二维平面填充了对角线的 Tensor。数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例** 

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph as dg
    import numpy as np
    
    diag_embed = np.random.randn(2, 3).astype('float32')
    # [[ 0.7545889 , -0.25074545,  0.5929117 ],
    #  [-0.6097662 , -0.01753256,  0.619769  ]]
    with dg.guard():
        data1 = fluid.layers.diag_embed(diag_embed)
        data1.numpy()
        # [[[ 0.7545889 ,  0.        ,  0.        ],
        #  [ 0.        , -0.25074545,  0.        ],
        #   [ 0.        ,  0.        ,  0.5929117 ]],

        # [[-0.6097662 ,  0.        ,  0.        ],
        #  [ 0.        , -0.01753256,  0.        ],
        #  [ 0.        ,  0.        ,  0.619769  ]]]

        data2 = fluid.layers.diag_embed(diag_embed, offset=-1, dim1=0, dim2=2)
        data2.numpy()
        # [[[ 0.        ,  0.        ,  0.        ,  0.        ],
        #   [ 0.7545889 ,  0.        ,  0.        ,  0.        ],
        #   [ 0.        , -0.25074545,  0.        ,  0.        ],
        #   [ 0.        ,  0.        ,  0.5929117 ,  0.        ]],
        #
        #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
        #   [-0.6097662 ,  0.        ,  0.        ,  0.        ],
        #   [ 0.        , -0.01753256,  0.        ,  0.        ],
        #   [ 0.        ,  0.        ,  0.619769  ,  0.        ]]]

        data3 = fluid.layers.diag_embed(diag_embed, offset=1, dim1=0, dim2=2)
        data3.numpy()
        # [[[ 0.        ,  0.7545889 ,  0.        ,  0.        ],
        #   [ 0.        , -0.6097662 ,  0.        ,  0.        ]],
        #
        #  [[ 0.        ,  0.        , -0.25074545,  0.        ],
        #   [ 0.        ,  0.        , -0.01753256,  0.        ]],
        #
        #  [[ 0.        ,  0.        ,  0.        ,  0.5929117 ],
        #   [ 0.        ,  0.        ,  0.        ,  0.619769  ]],
        #
        #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
        #   [ 0.        ,  0.        ,  0.        ,  0.        ]]]
