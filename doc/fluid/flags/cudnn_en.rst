==================
cudnn
==================


conv_workspace_size_limit
*******************************************
(since 0.13.0)

The workspace limit size in MB unit for choosing cuDNN convolution algorithms. The inner funciton of cuDNN obtain the fastest suited algorithm that fits within this memory limit. Usually, large workspace size may lead to choose faster algorithms, but significant increasing memory workspace. Users need to trade-off between memory and speed.

Values accepted
---------------
Uint64. The default value is 4096. That is to say, 4G memory workspace.

Example
-------
FLAGS_conv_workspace_size_limit=1024 set the workspace limit size for choosing cuDNN convolution algorithms to 1024MB.


cudnn_batchnorm_spatial_persistent
*******************************************
(since 1.4.0)

Indicates whether to use the new batch normalization mode CUDNN_BATCHNORM_SPATIAL_PERSISTENT function in batchnorm.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cudnn_batchnorm_spatial_persistent=True will enable the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.

Note
-------
This mode can be faster in some tasks because an optimized path will be selected for CUDNN_DATA_FLOAT and CUDNN_DATA_HALF data types. The reason we set it to False by default is that this mode may use scaled atomic integer reduction which may cause a numerical overflow for some input data range.


cudnn_deterministic
*******************************************
(since 0.13.0)

For one operation, cuDNN has several algorithms, some algorithm results are non-deterministic, like convolution algorithms. This flag is used for debugging. It indicates whether to choose the deterministic in cuDNN.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cudnn_deterministic=True will choose the deterministic in cuDNN.

Note
-------
Now this flag is enabled in cuDNN convolution and pooling operator. The deterministic algorithms may slower, so this flag is generally used for debugging.


cudnn_exhaustive_search
*******************************************
(since 1.2.0)

Whether to use exhaustive search method to choose convolution algorithms. There are two search methods, heuristic search and exhaustive search in cuDNN. The exhaustive search attempts all cuDNN algorithms to choose the fastest algorithm. This method is time-consuming, the choosed algorithm will be cached for the given layer specifications. Once the layer specifications (like batch size, feature map size) are changed, it will search again.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cudnn_exhaustive_search=True will use exhaustive search method to choose convolution algorithms.
