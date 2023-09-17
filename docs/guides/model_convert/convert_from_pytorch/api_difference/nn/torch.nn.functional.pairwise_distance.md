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
| logits | x            | 仅参数名不同                         |
| tau    | temperature  | 仅参数名不同                        |
| hard   | hard             | 参数完全相同 |
| eps    |              | Paddle无此参数，一般对网络训练结果影响不大，可直接删除                       |
| dim    |axis| 仅参数名不同                         |

