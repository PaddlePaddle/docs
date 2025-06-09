## [ torch 参数更多 ]torch.cholesky_inverse

### [torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html?highlight=cholesky_inverse#torch.cholesky_inverse)

```python
torch.cholesky_inverse(input, upper=False, out=None)
```

### [paddle.linalg.cholesky_inverse](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/cholesky_inverse_cn.html#cholesky-inverse)

```python
paddle.linalg.cholesky_inverse(x, upper=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
| upper   | upper        | 指示是否返回上三角矩阵或下三角矩阵。  |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

```python
# PyTorch 写法
torch.cholesky_inverse(input, out=output)

# Paddle 写法
paddle.assign(paddle.linalg.cholesky_inverse(input), output=output)
```
