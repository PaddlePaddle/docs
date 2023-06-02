.. _cn_api_tensor_linalg_cdist:

cdist
-------------------------------

.. py:function:: paddle.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", name=None)

计算两组输入集合中每对之间的 p 范数距离。

当 :math:`p \in (0, \infty)` 时，该函数等同于 `scipy.spatial.distance.cdist(input,'minkowski', p=p)` ；
当 :math:`p = 0` 时，等同于 `scipy.spatial.distance.cdist(input, 'hamming') * M` ；
当 :math:`p = \infty` 时，最接近的是 `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())` 。

参数
::::::::::::

  - **x** (Tensor) - 形状为 :math:`B \times P \times M` 的 Tensor。
  - **y** (Tensor) - 形状为 :math:`B \times R \times M` 的 Tensor。
  - **p** (float, 可选) - 计算每个向量对之间的 p 范数距离的值。默认值为 :math:`2.0`。
  - **compute_mode** (str, 可选) - 选择计算模式。

    - ``use_mm_for_euclid_dist_if_necessary``: 对于 p = 2.0 且 P > 25, R > 25 ，如果可能，将使用矩阵乘法计算欧氏距离。
    - ``use_mm_for_euclid_dist``: 对于 p = 2.0 ，使用矩阵乘法计算欧几里得距离。
    - ``use_loop_for_euclid_dist``: 不使用矩阵乘法计算欧几里得距离。

    默认值为 ``use_mm_for_euclid_dist_if_necessary``。
  - **name** (str, 可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，`dtype` 与输入张量相同。
如果 x 的形状为 :math:`B \times P \times M`，y 的形状为 :math:`B \times R \times M`，则输出的形状为 :math:`B \times P \times R`。

代码示例
::::::::::::

COPY-FROM: paddle.cdist
