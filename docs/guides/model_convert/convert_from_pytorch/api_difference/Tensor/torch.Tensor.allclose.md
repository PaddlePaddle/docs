## [ 返回参数类型不一致 ]torch.Tensor.allclose

### [torch.Tensor.allclose](https://pytorch.org/docs/stable/generated/torch.Tensor.allclose.html)

```python
torch.Tensor.allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False)
```

### [paddle.Tensor.allclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#allclose-y-rtol-1e-05-atol-1e-08-equal-nan-false-name-none)

```python
paddle.Tensor.allclose(y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，返回类型不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注 |
| --------- | ------------ | -- |
| other     | y            | 输入 Tensor，仅参数名不一致。 |
| rtol      | rtol         | 相对容差。 |
| atol      | atol         | 绝对容差。 |
| equal_nan | equal_nan    | 如果设置为 True，则两个 NaN 数值将被视为相等。 |
| 返回值    | 返回值        | PyTorch 返回值为标量， Paddle 返回值 0D Tensor。|

### 转写示例

#### 返回值

```python
# PyTorch 写法
x.allclose(y)

# Paddle 写法
x.allclose(y).item()
```
