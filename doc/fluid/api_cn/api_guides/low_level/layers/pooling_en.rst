.. _api_guide_pool_en:

#####
Pool
#####

Pooling is to downsample the input features and reduce overfitting. Reducing overfitting is the result of reducing the output size, which also reduces the number of parameters in subsequent layers.

Pooling usually only takes the feature map of the previous layer as input, and some parameters are needed to determine the specific operation of the pooling. In PaddlePaddle, we also choose the specific pooling method by setting the size, method, step, whether to pool globally, whether to use cudnn, whether to use ceil function to calculate output and other parameters.
PaddlePaddle has two-dimensional (pool2d), three-dimensional convolution (pool3d), RoI pooling (roi_pool) for sequence-length image features, and sequence pooling (sequence_pool) for sequences, as well as the reverse process of pooling calculations. The following describes the 2D/3D pooling, and the RoI pooling, and then introduces the sequence pooling.

--------------

1. pool2d/pool3d
------------------------

- ``input`` : The pooling operation receives any \``Tensor that conforms to the layout: \ ``N(batch size)* C(channel size) * H(height) * W(width)``\ format `\ type as input.

- ``pool_size``\ : Used to determine the size of the pooling \ ``filter``\, which is how much data is pooled into a single value.

- ``num_channels``\ : is used to determine the number of \ ``channel``\ entered. If no parameter is set or is set to \ ``None``\, its actual value will be automatically set to the input \ ``channel ``\ Quantity.

- ``pooling_type``\ : Receive one of the two types of \ ``agg`` and \ ``max``\ as the pooling method. The default value is \ ``max``\. Where \ ``max``\ means maximum pooling, ie calculating the maximum value of the data in the pooled ``filter`` area as output; and \``avg``\ means averaging pooling, ie Calculate the average of the data in the pooled ` `filter`` area as output.

- ``pool_stride``\ : means the step size of the pooled \ ``filter``\ moving on the input feature map.

- ``pool_padding``\ : Used to determine the size of \ ``padding``\ in the pooling, \``padding``\ is used to pool the features of the feature edge, select a different \ ` The `pool_padding``\ size determines how much zero is added to the edge of the feature map. Thereby determining the extent to which the edge features are pooled.

- ``global_pooling``\ : means whether to use global pooling. Global pooling refers to pooling using \ ``filter``\ of the same size as the feature map. This process can also use average pooling or The maximum pooling is used as a pooling method. Global pooling is usually used to replace the fully connected layer to greatly reduce the parameters to prevent overfitting.

- The ``use_cudnn``\ : option allows you to choose whether or not to use cudnn to optimize the calculation pooling speed.

- ``ceil_mode``\ : Whether to use the ceil function to calculate the output height and width. \ ``ceil mode``\ means ceiling mode, which means that the edges of the feature map that are insufficient \ ``filter size``\ will be retained, separately calculated, or can be understood as supplementing the original data. The edge with a value of -NAN. The floor mode directly discards the side of the \``filter size``\. The specific calculation formula is as follows:
    
    - Non\ ``ceil_mode``\ Under: \ ``Output size = (input size - filter size + 2 * padding) / stride (step size) + 1``
    
    - ``ceil_mode``\ 下:\ ``Output size = (input size - filter size + 2 * padding + stride - 1) / stride + 1``
    


Api Summary:

- :ref:`api_fluid_layers_pool2d`
- :ref:`api_fluid_layers_pool3d`


2. roi_pool
------------------

``roi_pool``\ is generally used to detect the network, and the input feature map is pooled to a specific size according to the candidate frame.

- ``rois``\ : Receive \ ``LoDTensor``\ type to indicate the Regions of Interest that needs to be pooled. For an explanation of RoI, please refer to \`Thesis <https://arxiv.org/abs/1506.01497>` __

- ``pooled_height`` and ``pooled_width``\ : where you can accept non-square pooled window sizes

- ``spatial_scale``\ : Used to set the scale of scaling the RoI and the original image. Note that the settings here require the user to calculate the actual scaling of the RoI and the original image.
 

Api Summary:

- :ref:`api_fluid_layers_roi_pool`


3. sequence_pool
--------------------

``sequence_pool``\ is an interface used to pool unequal sequences, it pools the features of all time steps of each instance, it also supports
One of the four types of ``average``, ``sum``, ``sqrt`` and \``max``\ is used as a pooling method. among them:

- ``average``\ is the result of summing the data in each time step and taking the average as the pooling result.

- ``sum``\ is the result of pooling the data in each time step as a pooling result.

- ``sqrt``\ is the result of summing the data in each time step and taking the square root as the pooling result.

- ``max``\ is the result of taking the maximum value for each time step as the pooling result.

Api Summary:

- :ref:`api_fluid_layers_sequence_pool`