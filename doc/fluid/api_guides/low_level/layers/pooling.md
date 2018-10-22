# 池化

减小输出大小 和 降低过拟合。降低过拟合是减小输出大小的结果，它同样也减少了后续层中的参数的数量

***
### `sequence_pool`
`sequence_pool`是一个用作进行序列池化的接口，他将每一个实例的全部time-step的特征进行池化，通常用在输入的上层。          

该接口签名：


	def sequence_pool(input, pool_type):

其中： 

- `input` : 接收任何`LoDTensor`类型作为输入
- `pooling_type ` : 接收`average`, `sum`, `sqrt` 和 `max`4种类型之一作为pooling的方式

使用时可仿照以下形式：

	x = fluid.layers.data(name='x', shape=[7, 1],dtype='float32', lod_level=1)
	avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')



### `pool2d`

`pool2d`是一个用来执行通用的2维池化的接口。          

该接口签名：
	
	def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None):
其中： 

- `input` : 接收任何符合`N（batch size）* C(channel size) * H(height) * W(width)`格式的`Tensor`类型作为输入
- `pool_size` : 接收`int`类型值来确定`pooling window`的长度，`pooling window`的大小将为该值的平方，，默认值为`-1`
- `num_channels` : 接收`int`类型的只来标示输入的`channel`数量，如果未设置参数或设置为`None`，其实际值将自动设置为输入的`channel`数量
- `pooling_type ` : 接收`avg`和 `max`2种类型之一作为pooling的方式
- `pool_stride` : 接收`int`类型作为输入来确定pooling步长的大小，默认值为`1`
- `pool_padding`  : 接收`int`类型输入来确定pooling时padding的大小. 默认值为`0`
- `global_pooling ` : 接收`bool`类型作为输入来确定是否使用全局池化，即将整个特征图池化
- `use_cudnn` : 接收`bool`类型值来确定是否使用`cudnn kernel`.  
- `ceil_mode` : 是否使用ceil函数计算输出高度和宽度。True是默认值。 如果设置为False，则使用`floor`功能。
- `name` : 接收`string`类型输入来设定输出的名字


使用时可仿照以下形式：
	
	data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
   	pool2d = fluid.layers.pool2d(
                            input=data,
                            pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)

### `pool3d `

`pool3d `是一个用来执行通用的3维池化的接口。              

该接口签名：
	
	def pool3d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None):
其中：

- `input` : 接收任何符合`N（batch size）* C(channel size) * H(height) * W(width)`格式的`Tensor`类型作为输入
- `pool_size` : 接收`int`类型值来确定`pooling window`的长度，`pooling window`的大小将为该值的平方，，默认值为`-1`
- `num_channels` : 接收`int`类型的只来标示输入的`channel`数量，如果未设置参数或设置为`None`，其实际值将自动设置为输入的`channel`数量
- `pooling_type ` : 接收`avg`和 `max`2种类型之一作为pooling的方式，默认值为`max`
- `pool_stride` : 接收`int`类型作为输入来确定pooling步长的大小，默认值为`1`
- `pool_padding`  : 接收`int`类型输入来确定pooling时padding的大小. 默认值为`0`
- `global_pooling ` : 接收`bool`类型作为输入来确定是否使用全局池化，即将整个特征图池化
- `use_cudnn` : 接收`bool`类型值来确定是否使用`cudnn kernel`.  
- `ceil_mode` : 是否使用ceil函数计算输出高度和宽度。True是默认值。 如果设置为False，则使用`floor`功能。
- `name` : 接收`string`类型输入来设定输出的名字        

使用时可仿照以下形式：
	
	data = fluid.layers.data(
              name='data', shape=[3, 3，32, 32], dtype='float32')
   	pool3d = fluid.layers.pool3d(
                            input=data,
                            pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)



### `roi_pool`
`roi_pool`是一个在`Fast R-CNN`中使用，用来从最后一个feature map中提取ROI的特征图。                 

该接口签名：  

	
	def roi_pool(input, 
					rois, 
					pooled_height=1, 
					pooled_width=1, 
					spatial_scale=1.0):  

其中：	

- `input`: 接收任何`LoDTensor`类型做为输入
- `rois` : 接收`LoDTensor`类型来表示需要池化的 Regions of Interest
- `pooled_height` : 接收`int`类型作为池化filter的高, 默认值为 1
- `pooled_width` : 接收`int`类型作为池化filter的宽, 默认值为 1
- `spatial_scale` : 接收`float`类型输入作为缩放ROI空间尺度的比例，默认值为 1.0   

使用时可以仿照以下形式：

	pool_out = fluid.layers.roi_pool(input=x, rois=rois, 7, 7, 1.0)


