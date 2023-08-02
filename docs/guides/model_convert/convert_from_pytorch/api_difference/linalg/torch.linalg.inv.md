## [torch 参数更多]torch.linalg.inv

### [torch.linalg.inv](https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv)

```python
torch.linalg.inv(A, *, out=None)
```

### [paddle.linalg.inv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/inv_cn.html)

```python
paddle.linalg.inv(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ---------------------------------------------------- |
| A       | x            | 输入 Tensor，仅参数名不一致。                        |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.inv(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.inv(x), y)
```
