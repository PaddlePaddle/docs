## [ 组合替代实现 ]torch.kthvalue

### [torch.kthvalue](https://pytorch.org/docs/stable/generated/torch.kthvalue.html?highlight=kthvalue#torch.kthvalue)

```python
torch.kthvalue(input, k, dim=None, keepdim=False, out=None)
```
### 功能介绍
用于获取某一维度上第 k 大的数值，PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。
```python
import paddle

def kthvalue(input, k, dim=None, keepdim=False, out=None):
    values = paddle.sort(input, axis=dim)
    indices = paddle.argsort(input, axis=dim)
    values = paddle.slice(values, axes=[dim], starts=[k-1], ends=[k])
    indices = paddle.slice(indices, axes=[dim], starts=[k-1], ends=[k])
    if not keepdim:
        values = paddle.flatten(values)
        indices = paddle.flatten(indices)
    return values, indices
```
## [ 仅参数名不一致 ]torch.kthvalue
### [torch.kthvalue](https://pytorch.org/docs/stable/generated/torch.kthvalue.html?highlight=kthvalue#torch.kthvalue)

```python
torch.kthvalue(input,
               k,
               dim=None,
               keepdim=False,
               *,
               out=None)
```

### [paddle.kthvalue](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/kthvalue_cn.html)

```python
paddle.kthvalue(x,
                k,
                axis=None,
                keepdim=False,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                   |
| k         | k           | 表示需要沿轴查找的第 k 小值。                   |
| dim         | axis            | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。                   |
| keepdim         | keepdim            | 是否在输出 Tensor 中保留减小的维度。                   |
| out         | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写 。                   |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.kthvalue(x, 2, 1, out=y)

# Paddle 写法
paddle.assign(paddle.kthvalue(x, 2, 1), y)
```
