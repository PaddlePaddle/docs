
cudnn
==================


FLAGS_conv_workspace_size_limit
*******************************************
(始于0.13.0)

用于选择cuDNN卷积算法的工作区限制大小（单位为MB）。cuDNN的内部函数在这个内存限制范围内获得速度最快的匹配算法。通常，在较大的工作区内可以选择更快的算法，但同时也会显著增加内存空间。用户需要在内存和速度之间进行权衡。

取值范围
---------------
Uint64型，缺省值为512。即512MB显存工作区。

示例
-------
FLAGS_conv_workspace_size_limit=1024 - 将用于选择cuDNN卷积算法的工作区限制大小设置为1024MB。


FLAGS_cudnn_batchnorm_spatial_persistent
*******************************************
(始于1.4.0)

表示是否在batchnorm中使用新的批量标准化模式CUDNN_BATCHNORM_SPATIAL_PERSISTENT函数。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_batchnorm_spatial_persistent=True - 开启CUDNN_BATCHNORM_SPATIAL_PERSISTENT模式。

注意
-------
此模式在某些任务中可以更快，因为将为CUDNN_DATA_FLOAT和CUDNN_DATA_HALF数据类型选择优化路径。我们默认将其设置为False的原因是此模式可能使用原子整数缩减(scaled atomic integer reduction)而导致某些输入数据范围的数字溢出。


FLAGS_cudnn_deterministic
*******************************************
(始于0.13.0)

cuDNN对于同一操作有几种算法，一些算法结果是非确定性的，如卷积算法。该flag用于调试。它表示是否选择cuDNN中的确定性函数。 

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_deterministic=True - 选择cuDNN中的确定性函数。

注意
-------
现在，在cuDNN卷积和池化Operator中启用此flag。确定性算法速度可能较慢，因此该flag通常用于调试。


FLAGS_cudnn_exhaustive_search
*******************************************
(始于1.2.0)

表示是否使用穷举搜索方法来选择卷积算法。在cuDNN中有两种搜索方法，启发式搜索和穷举搜索。穷举搜索尝试所有cuDNN算法以选择其中最快的算法。此方法非常耗时，所选择的算法将针对给定的层规格进行缓存。 一旦更改了图层规格（如batch大小，feature map大小），它将再次搜索。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_cudnn_exhaustive_search=True - 使用穷举搜索方法来选择卷积算法。