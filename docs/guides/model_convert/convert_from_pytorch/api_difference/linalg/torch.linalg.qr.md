## [torch 参数更多]torch.linalg.qr

### [torch.linalg.qr](https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr)

```python
torch.linalg.qr(A, mode='reduced', *, out=None)
```

### [paddle.linalg.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/qr_cn.html)

```python
paddle.linalg.qr(x, mode='reduced', name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| A       | x            | 输入 Tensor，仅参数名不一致。                      |
| mode    | mode         | 控制正交三角分解的行为。                           |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.qr(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.qr(x), y)
```
