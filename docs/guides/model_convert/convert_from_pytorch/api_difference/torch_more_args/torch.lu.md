## [ torch 参数更多 ]torch.lu
### [torch.lu](https://docs.pytorch.org/docs/stable/generated/torch.lu.html#torch.lu)
```python
torch.lu(A, pivot=True, get_infos=False, out)
```

### [paddle.linalg.lu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_cn.html#paddle.linalg.lu)
```python
paddle.linalg.lu(x, pivot=True, get_infos=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  A          |  x             | 输入的 Tensor ，仅参数名不一致。                                     |
|  pivot     |  pivot         | 输入的 bool ，参数完全一致。                                     |
|  get_infos  |  get_infos     | 输入的 bool ，参数完全一致。                                     |
|  out        | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。             |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.lu(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), get_infos=True, out=(A_LU, pivots, info))

# Paddle 写法
A_LU, pivots, info = paddle.linalg.lu(paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), get_infos=True)
```
