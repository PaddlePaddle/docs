## [参数完全一致]torch.nn.functional.pairwise_distance

### [torch.nn.functional.pairwise_distance](https://pytorch.org/docs/stable/generated/torch.nn.functional.pairwise_distance.html)

```python
torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False)
```

### [paddle.nn.functional.pairwise_distance](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pairwise_distance_cn.html)

```python
paddle.nn.functional.pairwise_distance(x, y, p=2., epsilon=1e-6, keepdim=False, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                           |
|--------|--------------|------------------------------|
| x1 | x            | 仅参数名不同                         |
| x2    | y  | 仅参数名不同                        |
| p   | p             | 参数完全相同 |
| eps    |epsilon              | 仅参数名不同  |
| keepdim    |keepdim| 参数完全相同                         |

