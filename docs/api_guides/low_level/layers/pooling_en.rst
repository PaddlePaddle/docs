.. _api_guide_pool_en:

########
Pooling
########

Pooling is to downsample the input features and reduce overfitting. Reducing overfitting is the result of reducing the output size, which also reduces the number of parameters in subsequent layers.

Pooling usually only takes the feature maps of the previous layer as input, and some parameters are needed to determine the specific operation of the pooling. In PaddlePaddle, we also choose the specific pooling by setting parameters like the size, method, step, whether to pool globally, whether to use cudnn, whether to use ceil function to calculate output.
PaddlePaddle has two-dimensional (pool2d), three-dimensional convolution (pool3d), RoI pooling (roi_pool) for fixed-length image features, and sequence pooling (sequence_pool) for sequences, as well as the reverse(backward) process of pooling calculations. The following text describes the 2D/3D pooling, and the RoI pooling, and then the sequence pooling.

--------------

1. pool2d/pool3d
------------------------

- ``x`` : The pooling operation receives any ``Tensor`` that conforms to the layout: ``N(batch size)* C(channel size) * H(height) * W(width)`` format as input.

- ``kernel_size`` : It determines the size of the pooling kernel.

- ``stride`` : It determines the stride of the pooling operation.

- ``padding`` : It is used to determine the size of  ``padding`` in the pooling, ``padding`` is used to pool the features of the edges of feature maps. The ``pool_padding`` size determines how much zero is padded to the edge of the feature maps. Thereby it determines the extent to which the edge features are pooled.

- ``ceil_mode`` : Whether to use the ceil function to calculate the output height and width.  ``ceil mode`` means ceiling mode, which means that, in the feature map, the edge parts that are smaller than ``filter size`` will be retained, and separately calculated. It can be understood as supplementing the original data with edge with a value of -NAN. By contrast, The floor mode directly discards the edges smaller than the ``filter size``. The specific calculation formula is as follows:

  * Non ``ceil_mode`` :  ``Output size = (input size - filter size + 2 * padding) / stride (stride size) + 1``

  * ``ceil_mode`` : ``Output size = (input size - filter size + 2 * padding + stride - 1) / stride + 1``

- ``exclusive`` : It indicates whether to ignore padding values in average pooling mode. Defaults to True, meaning padding values are ignored.

- ``divisor_override`` : If specified, it will be used as the divisor. By default, it is not specified, and the divisor is automatically calculated based on ``kernel_size``.

- ``data_format`` : It represents the format of the input and output data, which can be either ``NCHW`` or ``NHWC``.

- ``return_mask`` : It indicates whether to return the max indices and the output (This parameter is only supported in the max pooling operation.).

related API:

- :ref:`api_paddle_nn_functional_avg_pool2d`
- :ref:`api_paddle_nn_functional_max_pool2d`
- :ref:`api_paddle_nn_functional_avg_pool3d`
- :ref:`api_paddle_nn_functional_max_pool3d`


2. roi_pool
------------------

``roi_pool`` is generally used in detection networks, and the input feature map is pooled to a specific size by the bounding box.

-  ``x`` : The input feature map, with a shape of (N, C, H, W).

- ``boxes`` : It receives ``Tensor`` type to indicate the Regions of Interest that needs to be pooled. For an explanation of RoI, please refer to `Paper <https://arxiv.org/abs/1506.01497>`__

-  ``boxes_num`` : The number of boxes contained in each image of the batch.

-  ``output_size`` : The size (H, W) of the output after pooling, with a data type of int32.

- ``spatial_scale`` : Used to set the scale of scaling the RoI and the original image. Note that the settings here require the user to manually calculate the actual scaling of the RoI and the original image.


related API:

- :ref:`api_paddle_vision_ops_roi_pool`


3. sequence_pool
--------------------

``sequence_pool`` is an interface used to pool variable-length sequences. It pools the features of all time steps of each instance, and also supports
one of  ``average``, ``sum``, ``sqrt``, ``max``, ``last`` and ``first`` to be used as the pooling method. Specifically:

- ``average`` sums up the data in each time step and takes its average as the pooling result.

- ``sum`` take the sum of the data in each time step as pooling result.

- ``sqrt`` sums the data in each time step and takes its square root as the pooling result.

- ``max`` takes the maximum value for each time step as the pooling result.

- ``last`` takes the final value for each time step as the pooling result.

- ``first`` takes the first value for each time step as the pooling result.

related API:

- :ref:`api_paddle_static_nn_sequence_pool`
