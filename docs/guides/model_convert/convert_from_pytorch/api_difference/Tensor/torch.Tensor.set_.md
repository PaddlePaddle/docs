## [ 输入参数用法不一致 ]torch.Tensor.set_
### [torch.Tensor.set_](https://pytorch.org/docs/stable/generated/torch.Tensor.set_.html)

```python
torch.Tensor.set_(source=None, storage_offset=0, size=None, stride=None)
```

### [paddle.Tensor.set_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#set-source-none-shape-none-stride-none-offset-0-name-none)

```python
paddle.Tensor.set_(source=None, shape=None, stride=None, offset=0, name=None)
```

其中 PyTorch 的 `storage_offset` 与 Paddle 的 `offset` 用法不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| source         | source      | 设置的目标 Tensor。                    |
| storage_offset | offset      | 表示距离内存数据起始位置的偏移量。Pytorch 是指偏移数字的个数，而 Paddle 是指偏移数字对应存储位置的偏移量（以 byte 为单位），需要转写。  |
| size           | shape       | 设置的目标形状, 仅参数名不一致。        |
| stride         | stride      | 设置的目标步长。                       |


### 转写示例

#### storage_offset 参数：float32 偏移量设置
``` python
# PyTorch 写法:
x = torch.tensor([[1., 2.]], dtype=torch.float32)
src = torch.tensor([11., 22., 33., 44., 55.], dtype=torch.float32)
offset_num = 2
x.set_(src, storage_offset=offset_num, size=[3], stride=[1])

# Paddle 写法:
x = paddle.to_tensor([[1., 2.]], dtype=paddle.float32)
src = paddle.to_tensor([11., 22., 33., 44., 55.], dtype=paddle.float32)
offset_num = 2
x.set_(src, shape=[3], stride=[1], offset=offset_num*4)
```

#### storage_offset 参数：float64 偏移量设置
``` python
# PyTorch 写法:
x = torch.tensor([[1., 2.]], dtype=torch.float64)
src = torch.tensor([11., 22., 33., 44., 55.], dtype=torch.float64)
offset_num = 2
x.set_(src, storage_offset=offset_num, size=[3], stride=[1])

# Paddle 写法:
x = paddle.to_tensor([[1., 2.]], dtype=paddle.float64)
src = paddle.to_tensor([11., 22., 33., 44., 55.], dtype=paddle.float64)
offset_num = 2
x.set_(src, shape=[3], stride=[1], offset=offset_num*8)
```
