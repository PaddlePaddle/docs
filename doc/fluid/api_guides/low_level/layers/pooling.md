# 池化

减小输出大小 和 降低过拟合。降低过拟合是减小输出大小的结果，它同样也减少了后续层中的参数的数量

***
##`sequence_pool`
`sequence_pool`是一个用作进行序列池化的接口，他将每一个实例的全部time-step的特征进行池化，通常用在输入的上层。          


## `pool2d`

`pool2d`是一个用来执行通用的对于2维`feature map`进行池化的接口。          


## `pool3d `

`pool3d `是一个用来执行通用的对于3维`feature map`进行池化的接口。              


## `roi_pool`
`roi_pool`是一个在`Fast R-CNN`中使用，用来从最后一个`feature map`中提取`ROI`（Region Of Interest）的特征图的池化接口。 