## [ torch 参数更多 ]torch.linalg.householder_product

### [torch.linalg.householder_product](https://pytorch.org/docs/stable/generated/torch.linalg.householder_product.html#torch.linalg.householder_product)

```python
torch.linalg.householder_product(input, tau, *, out=None)
```

### [paddle.linalg.householder_product](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/householder_product_cn.html#householder-product)

```python
paddle.linalg.householder_product(x, tau, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                           |
| ------- | ------------ | ---------------------------------------------- |
| input   | x            | 表示输入的 Tensor，仅参数名不一致。            |
| tau     | tau          | 表示输入的 Tensor。                            |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.householder_product(x, tau, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.householder_product(x, tau), y)
```
