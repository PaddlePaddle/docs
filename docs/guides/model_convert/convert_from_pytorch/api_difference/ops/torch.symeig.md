## [ torch 参数更多 ] torch.symeig

### [torch.symeig](https://pytorch.org/docs/stable/generated/torch.symeig.html?highlight=torch+symeig#torch.symeig)

```python
torch.symeig(input, eigenvectors=False, upper=True, *, out=None)
```

### [paddle.linalg.eigh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/eigh_cn.html#eigh)

```python
paddle.linalg.eigh(x, UPLO='L', name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x            | 输入的对称 Tensor，仅参数名不一致。                           |
| eigenvectors   | -            | 表示是否计算特征向量。Paddle 无此参数，暂无转写方式。      |
| upper          | UPLO            | 表示计算上三角或者下三角矩阵。 需进行转写。                          |
| out          | -            | 表示输出的 Tensor 元组， Paddle 无此参数，需要转写。                           |

### 转写示例
#### upper：表示计算上三角或者下三角矩阵
```python
# Pytorch 写法
e, v = torch.symeig(x, upper = False)

# Paddle 写法
e, v = paddle.linalg.eigh(x, UPLO = 'L')
```

#### out：指定输出
```python
# Pytorch 写法
torch.symeig(x, out=(e, v) )

# Paddle 写法
e, v = paddle.linalg.eigh(x)
```
