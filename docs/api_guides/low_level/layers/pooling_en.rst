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

- ``input`` : The pooling operation receives any ``Tensor`` that conforms to the layout: ``N(batch size)* C(channel size) * H(height) * W(width)`` format as input.

- ``pool_size`` : It is used to determine the size of the pooling  ``filter``, which determines the size of data to be pooled into a single value.

- ``num_channels`` : It is used to determine the number of ``channel`` of input. If it is not set or is set to ``None``, its actual value will be automatically set to the ``channel`` quantity of input.

- ``pool_type`` : It receives one of ``agg`` and ``max`` as the pooling method. The default value is  ``max`` . ``max`` means maximum pooling, i.e. calculating the maximum value of the data in the pooled ``filter`` area as output; and ``avg`` means averaging pooling, i.e. calculating the average of the data in the pooled  ``filter`` area as output.

- ``pool_stride`` : It is the stride size in which the pooling ``filter`` moves on the input feature map.

- ``pool_padding`` : It is used to determine the size of  ``padding`` in the pooling, ``padding`` is used to pool the features of the edges of feature maps. The ``pool_padding`` size determines how much zero is padded to the edge of the feature maps. Thereby it determines the extent to which the edge features are pooled.

- ``global_pooling`` : It Means whether to use global pooling. Global pooling refers to pooling using  ``filter`` of the same size as the feature map. This process can also use average pooling or the maximum pooling as the pooling method. Global pooling is usually used to replace the fully connected layer to greatly reduce the parameters to prevent overfitting.

- The ``use_cudnn`` : This option allows you to choose whether or not to use cudnn to accelerate pooling.

- ``ceil_mode`` : Whether to use the ceil function to calculate the output height and width.  ``ceil mode`` means ceiling mode, which means that, in the feature map, the edge parts that are smaller than ``filter size`` will be retained, and separately calculated. It can be understood as supplementing the original data with edge with a value of -NAN. By contrast, The floor mode directly discards the edges smaller than the ``filter size``. The specific calculation formula is as follows:

  * Non ``ceil_mode`` :  ``Output size = (input size - filter size + 2 * padding) / stride (stride size) + 1``

  * ``ceil_mode`` : ``Output size = (input size - filter size + 2 * padding + stride - 1) / stride + 1``



related API:

- :ref:`api_fluid_layers_pool2d`
- :ref:`api_fluid_layers_pool3d`


2. roi_pool
------------------

``roi_pool`` is generally used in detection networks, and the input feature map is pooled to a specific size by the bounding box.

- ``rois`` : It receives ``LoDTensor`` type to indicate the Regions of Interest that needs to be pooled. For an explanation of RoI, please refer to `Paper <https://arxiv.org/abs/1506.01497>`__

- ``pooled_height`` and ``pooled_width`` : accept non-square pooling box sizes

- ``spatial_scale`` : Used to set the scale of scaling the RoI and the original image. Note that the settings here require the user to manually calculate the actual scaling of the RoI and the original image.


related API:

- :ref:`api_fluid_layers_roi_pool`


3. sequence_pool
--------------------

``sequence_pool`` is an interface used to pool variable-length sequences. It pools the features of all time steps of each instance, and also supports
one of  ``average``, ``sum``, ``sqrt`` and ``max`` to be used as the pooling method. Specifically:

- ``average`` sums up the data in each time step and takes its average as the pooling result.

- ``sum`` take the sum of the data in each time step as pooling result.

- ``sqrt`` sums the data in each time step and takes its square root as the pooling result.

- ``max`` takes the maximum value for each time step as the pooling result.

related API:

- :ref:`api_fluid_layers_sequence_pool`
