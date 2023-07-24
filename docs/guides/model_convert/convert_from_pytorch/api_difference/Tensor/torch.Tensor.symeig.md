## [torch 参数更多]torch.Tensor.symeig

### [torch.Tensor.symeig](https://pytorch.org/docs/stable/generated/torch.Tensor.symeig.html#torch.Tensor.symeig)

```python
torch.Tensor.symeig(eigenvectors=False, upper=True)
```

### [paddle.linalg.eigh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/eigh_cn.html)

```python
paddle.linalg.eigh(x, UPLO='L', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| -            | x            | 表示输入的 Tensor 。                                         |
| eigenvectors | -            | 表示是否计算 eigenvectors，Paddle 无此参数，暂无转写方式。 |
| upper        | UPLO         | 表示计算上三角或者下三角矩阵，PyTorch 取值 bool 类型，Paddle 取值 L, U。 |

### 转写示例

#### upper：表示计算上三角或者下三角矩阵
```python
# Pytorch 写法
e, v = x.symeig(upper = False)

# Paddle 写法
e, v = paddle.linalg.eigh(x, UPLO = 'L')
```
