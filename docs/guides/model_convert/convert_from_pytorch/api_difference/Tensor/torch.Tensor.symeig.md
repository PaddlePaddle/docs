## [torch 参数更多]torch.Tensor.symeig

### [torch.Tensor.symeig](https://pytorch.org/docs/stable/generated/torch.Tensor.symeig.html#torch.Tensor.symeig)

```python
# pytorch1.9 以上版本不支持
torch.Tensor.symeig(eigenvectors=False, upper=True)
```

### [paddle.linalg.eigh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/eigh_cn.html)

```python
# eigenvectors 为 True
paddle.linalg.eigh(x, UPLO='L', name=None)

# eigenvectors 为 False
paddle.linalg.eigvalsh(x, UPLO='L', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| -            | x            | 表示输入的 Tensor 。                                         |
| eigenvectors | -            | 表示是否计算 eigenvectors，Paddle 无此参数，需要转写。 |
| upper        | UPLO         | 表示计算上三角或者下三角矩阵，PyTorch 取值 bool 类型，Paddle 取值 L, U。 |

### 转写示例

#### eigenvectors：表示是否计算特征向量
```python
# PyTorch 写法，eigenvectors 为 False
e, _ = x.symeig(eigenvectors=False)

# Paddle 写法
e = paddle.linalg.eigvalsh(x)

# PyTorch 写法，eigenvectors 为 True
e, v = x.symeig(eigenvectors=True)

# Paddle 写法
e, v = paddle.linalg.eigh(x)
```

#### upper：表示计算上三角或者下三角矩阵
```python
# PyTorch 写法
e, v = x.symeig(upper = False)

# Paddle 写法
e, v = paddle.linalg.eigh(x, UPLO = 'L')
```
