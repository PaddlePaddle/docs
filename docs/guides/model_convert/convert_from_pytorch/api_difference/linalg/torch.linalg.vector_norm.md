## [torch 参数更多]torch.linalg.vector_norm

### [torch.linalg.vector_norm](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm)

```python
torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None)
```

### [paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/norm_cn.html)

```python
paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                            |
| ------- | ------------ | ------------------------------------------------------------------------------- |
| x       | x            | 输入 Tensor。                                                                   |
| ord     | p            | 范数的种类。参数不一致。Pytorch 支持负实数的范数，Paddle 不支持，暂无转写方式。 |
| dim     | axis         | 使用范数计算的轴 ，仅参数名不一致。                                             |
| keepdim | keepdim      | 是否在输出的 Tensor 中保留和输入一样的维度。                                    |
| dtype   | -            | 表示输出 Tensor 的数据类型， Paddle 无此参数，需要转写。                        |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。                                |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.vector_norm(x, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.norm(x), y)
```

#### dtype：表示输出 Tensor 的数据类型

```python
# Pytorch 写法
torch.linalg.vector_norm(x, dtype=torch.float64)

# Paddle 写法
paddle.linalg.norm(x.astype(paddle.float64))
```
