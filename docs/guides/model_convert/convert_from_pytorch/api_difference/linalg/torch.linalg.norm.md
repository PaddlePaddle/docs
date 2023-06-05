## [ torch 参数更多 ]torch.linalg.norm

### [torch.linalg.norm](https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm)

```python
torch.linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None)
```

### [paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/norm_cn.html#norm)

```python
paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None)
```

Pytorch 支持更多的参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input | x         | 表示输入的一个 tensor 列表 ，仅参数名不一致。                    |
| ord | p         | 范数的种类。参数不一致。Pytorch 支持负实数的范数，Paddle 暂不支持，暂无转写方式。                   |
| dim | axis         | 使用范数计算的轴 ，仅参数名不一致。                    |
| keepdim | keepdim         | 是否在输出的 Tensor 中保留和输入一样的维度。                    |
| out       | -       | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。 |
| dtype       | -       | 表示输出 Tensor 的数据类型， Paddle 无此参数，需要进行转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.norm(x, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.norm(x), y)
```

#### dtype：表示输出 Tensor 的数据类型

```python
# Pytorch 写法
torch.linalg.norm(x, dtype=torch.float64)

# Paddle 写法
paddle.linalg.norm(x).astype(paddle.float64)
```
