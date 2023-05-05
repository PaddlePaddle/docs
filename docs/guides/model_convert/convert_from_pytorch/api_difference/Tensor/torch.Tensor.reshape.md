## [部分参数不一致]torch.Tensor.reshape

### [torch.Tensor.reshape](https://pytorch.org/docs/1.13/generated/torch.Tensor.reshape.html)

'''python
    torch.Tensor.reshape(*shape) 
'''

### [paddle.Tensor.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#reshape-shape-name-none)

'''python
    paddle.Tensor.reshape(shape, name=None)
'''

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *shape | shape | torch的 *shape 既可以接收list也可接收可变参数。|

# 代码转写

```python
    # pytorch
    x = torch.randn(2, 3, 5)
    x_reshape = x.reshape(6,5)
    x_reshape_tuple = x.reshape((6,5))
    x_reshape_list = x.reshape([6,5])

    # paddle
    x = paddle.randn([2, 3, 5])
    x_tile_tuple = x.tile((6,5))
    x_tile_list = x.tile([6,5])
    x_tile_tensor = x.tile(paddle.to_tensor([6,5], dtype='int32'))
```