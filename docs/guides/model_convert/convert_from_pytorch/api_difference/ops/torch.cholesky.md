## [ torch 参数更多]torch.cholesky

### [torch.cholesky](https://pytorch.org/docs/stable/generated/torch.cholesky.html?highlight=cholesky#torch.cholesky)

```python
torch.cholesky(input,upper=False,*,out=None)
```

### [paddle.linalg.cholesky](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cholesky_cn.html)

```python
paddle.linalg.cholesky(x,upper=False,name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| input | x | 表示输入参数为多维 Tensor ，它的维度应该为 [*, M, N]，其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵，仅参数名不一致。 |
| upper | upper | 表示是否返回上三角矩阵或下三角矩阵。 |
| out | - | 表示输出的 Tensor， Paddle 无此参数，需要进行转写。 |

### 转写示例

#### out：输出的 Tensor

```python
# Pytorch 写法
torch.cholesky(x, upper=False, out=output)


# Paddle 写法
paddle.assign(paddle.linalg.cholesky(x, upper=False), output)
```
