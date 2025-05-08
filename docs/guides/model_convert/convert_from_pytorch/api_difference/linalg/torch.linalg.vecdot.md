## [torch 参数更多]torch.linalg.vecdot

### [torch.linalg.vecdot](https://pytorch.org/docs/stable/generated/torch.linalg.vecdot.html)

```python
torch.linalg.vecdot(x, y, *, dim=-1, out=None)
```

### [paddle.linalg.vecdot](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py#L1881)

```python
paddle.linalg.vecdot(x, y, axis=-1, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                            |
| ------- | ------------ | ------------------------------------------------------------------------------- |
| x       | x            | 输入 Tensor。                                                                   |
| y     | y            | 输入 Tensor。 |
| dim     | axis         | 计算向量点积的维度 ，仅参数名不一致。                                             |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。                                |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.linalg.vecdot(x=a, y=b, out=out)

# Paddle 写法
paddle.assign(paddle.linalg.vecdot(x=a, y=b), output=out)
```
