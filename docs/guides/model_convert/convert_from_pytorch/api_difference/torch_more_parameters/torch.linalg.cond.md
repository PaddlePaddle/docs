## [torch 参数更多]torch.linalg.cond

### [torch.linalg.cond](https://pytorch.org/docs/stable/generated/torch.linalg.cond.html#torch.linalg.cond)

```python
# PyTorch 文档有误，第一个参数名为 input
torch.linalg.cond(input, p=None, *, out=None)
```

### [paddle.linalg.cond](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cond_cn.html)

```python
paddle.linalg.cond(x, p=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
| ------- | ------------ | ---------------------------------------------------- |
| input       | x            | 输入 Tensor，仅参数名不一致。                        |
| p       | p            | 范数种类。                                           |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.cond(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.cond(x), y)
```
