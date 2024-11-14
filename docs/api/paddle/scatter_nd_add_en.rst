.. _en_api_paddle_scatter_nd_add:

scatter_nd_add
-------------------------------

.. py:function:: paddle.scatter_nd_add(x, index, updates, name=None)




Performs sparse addition on individual values or slices in a Tensor, resulting in an output Tensor.

:code:`x` is a Tensor with a dimension of :code:`R`. :code:`index` is a Tensor with a dimension of :code:`K`, which means that the shape of :code:`index` is :math:`[i_0, i_1, ..., i_{K-2}, Q]`, where :math:`Q \leq R`. :code:`updates` is a Tensor with a dimension of :math:`K - 1 + R - Q` and a shape of :math:`index.shape[:-1] + x.shape[index.shape[-1]:]`.

According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :code:`index`, it selects the corresponding :code:`updates` slice and adds it to the :code:`x` slice obtained by the last dimension of :code:`index`, resulting in the final output Tensor.

Examples:

::

        - Example 1:
            x = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          Result:

            output = [0, 22, 12, 14, 4, 5]

        - Example 2:
            x = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            x.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          Result:

            output = [[67, 19], [-16, -27]]


Parameters
::::::::::::

    - **x** (Tensor) - The input tensor, data type can be int32, int64, float32, float64.
    - **index** (Tensor) - The index tensor, data type must be non-negative int32 or non-negative int64. Its dimension :code:`index.ndim` must be greater than 1, and :code:`index.shape[-1] <= x.ndim`
    - **updates** (Tensor) - The update tensor, which must have the same data type as :code:`x`. Its shape must be :code:`index.shape[:-1] + x.shape[index.shape[-1]:]`.
    - **name** (str, optional) - For more details, refer to :ref:`api_guide_Name`. Generally, this does not need to be set. Default is None.

Returns
::::::::::::

A tensor with the same data type and shape as :code:`x`.

Example Code
::::::::::::


                    output = [0, 22, 12, 14, 4, 5]

**Explanation of Example 1**:

In this example, the scatter_nd_add function of Paddle performs sparse addition on the tensor `x`. The initial tensor `x` is `[0, 1, 2, 3, 4, 5]`. The `index` specifies the positions to be updated, and the values in `updates` are used to accumulate them. The scatter_nd_add function will add the corresponding values in `updates` to the specified positions in `x`, rather than replacing the original values. Finally, the output tensor is `[0, 22, 12, 14, 4, 5]`, achieving the cumulative update of specific elements in the tensor while keeping other elements unchanged.
    .. figure:: ../../images/api_legend/scatter_nd_add.png
       :width: 700
       :alt: Example 1 Illustration
       :align: center
