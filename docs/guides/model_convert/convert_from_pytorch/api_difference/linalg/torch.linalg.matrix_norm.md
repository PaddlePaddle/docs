## [torch 参数更多]torch.linalg.matrix_norm

### [torch.linalg.matrix_norm](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm)

```python
torch.linalg.matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None)
```

### [paddle.linalg.matrix_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_norm_cn.html)

```python
paddle.linalg.matrix_norm(x, p='fro', axis=[-2,-1], keepdim=False, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                            |
| ------- | ------------ | ------------------------------------------------------------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。                                                   |
| ord     | p            | 范数的种类，仅参数名不一致。 |
| dim     | axis         | 使用范数计算的轴 ，仅参数名不一致。                                             |
| keepdim | keepdim      | 是否在输出的 Tensor 中保留和输入一样的维度。                                    |
| dtype   | -            | 表示输出 Tensor 的数据类型， Paddle 无此参数，需要转写。                        |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。                                |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.linalg.matrix_norm(x, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.matrix_norm(x), y)
```

#### dtype：表示输出 Tensor 的数据类型

```python
# PyTorch 写法
torch.linalg.matrix_norm(x, dtype=torch.float64)

# Paddle 写法
paddle.linalg.matrix_norm(x.astype(paddle.float64))
```
