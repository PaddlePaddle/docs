## [torch 参数更多]torch.nn.functional.gumbel_softmax

### [torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html?highlight=gumbel_softmax#torch.nn.functional.gumbel_softmax)

```python
torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=- 1)
```

### [paddle.nn.functional.gumbel_softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gumbel_softmax_cn.html#gumbel-softmax)

```python
paddle.nn.functional.gumbel_softmax(x,temperature=1.0,hard=False,axis=-1,name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                           |
|--------|--------------|------------------------------|
| logits | x            | 输入参数                         |
| tau    | temperature  | 0到1的值                        |
| hard   |              | 为True时，将进行one-hot编码，默认为False |
| eps    |              | 计算的误差值                       |
| dim    |axis| 计算的维度                        |

### 转写示例

```python
# Pytorch 写法
import torch
logits = torch.randn(20, 32)
F.gumbel_softmax(logits, tau=1)


# Paddle 写法
 
import paddle
x = paddle.randn(shape=[20, 32])
paddle.nn.functional.gumbel_softmax(x, temperature=1)


```
