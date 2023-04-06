## [ 组合替代实现 ]torch.rand_like

### [torch.rand_like](https://pytorch.org/docs/master/generated/torch.rand_like.html#torch.rand_like)
```python
torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

###  功能介绍
返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。

### 参数介绍
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | -            | 表示输入的 Tensor                                   |
| dtype         | -            | 表示数据类型。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，需要进行转写。                   |
| requires_grad | stop_gradient            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数，需要进行转写。 |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。               |

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

```python
import paddle

def rand_like(input, dtype=None, requires_grad=False):
    if dtype is not None:
        out = paddle.rand(shape=input.shape, dtype=dtype)
    else:
        out = paddle.rand(shape=input.shape, dtype=input.dtype)
    out.stop_gradient = not requires_grad
    return out
```
