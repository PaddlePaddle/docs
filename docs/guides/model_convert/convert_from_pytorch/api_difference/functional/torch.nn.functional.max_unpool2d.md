## [ 参数不一致 ]torch.nn.functional.max_unpool2d

### [torch.nn.functional.max_unpool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_unpool2d.html?highlight=max_unpool2d#torch.nn.functional.max_unpool2d)

```python
torch.nn.functional.max_unpool2d(input,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 output_size=None)
```

### [paddle.nn.functional.max_unpool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/max_unpool2d_cn.html)

```python
paddle.nn.functional.max_unpool2d(x,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 data_format='NCHW',
                                 output_size=None,
                                 name=None)
```

其中 Paddle 和 PyTorch 的`indices`参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| indices           | indices           | 表示索引下标，PyTorch 数据类型为 int64, Paddle 数据类型为 int32，需要转写。                 |
| kernel_size           | kernel_size           | 表示滑动窗口大小。               |
| stride           | stride           | 表示步长。               |
| padding           | padding           | 表示填充大小。               |
| output_size           | output_size           | 表示目标输出尺寸。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |

### 转写示例
#### indices：索引下标
```python
# PyTorch 写法
result = F.max_unpool2d(x, indices, kernel_size=2, padding=0)

# Paddle 写法
result = F.max_unpool2d(x, indices.astype('int32'), kernel_size=2, padding=0)
```
