## [ torch 参数更多 ]torch.norm

### [torch.norm](https://pytorch.org/docs/stable/generated/torch.norm.html)

```python
torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
```

### [paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/norm_cn.html#norm)

```python
paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| p       | p            | 范数(ord)的种类。 |
| dim     | axis         | 使用范数计算的轴，仅参数名不一致。 |
| keepdim | keepdim      | 是否在输出的 Tensor 中保留和输入一样的维度。 |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。          |
| dtype   | -            | 表示输出 Tensor 的数据类型， Paddle 无此参数，需要转写。                        |

### 转写示例

#### out 参数：指定输出
``` python
# PyTorch 写法:
torch.norm(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.norm(x) , y)
```

#### dtype：表示输出 Tensor 的数据类型

```python
# PyTorch 写法
torch.norm(x, dtype=torch.float64)

# Paddle 写法
paddle.linalg.norm(x.astype(paddle.float64))
```
