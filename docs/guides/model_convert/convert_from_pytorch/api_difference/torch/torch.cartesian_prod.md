## [输入参数类型不一致]torch.cartesian_prod

### [torch.cartesian_prod](https://pytorch.org/docs/stable/generated/torch.cartesian_prod.html#torch-cartesian-prod)

```python
torch.cartesian_prod(*tensors)
```

### [paddle.cartesian_prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cartesian_prod_cn.html)

```python
paddle.cartesian_prod(x, name=None)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                         |
| -------- | ------------ | ------------------------------------------------------------ |
| *tensors | x            | 一组输入 Tensor ， PyTorch 参数 tensors 为可变参, Paddle 参数 x 为 list(Tensor) 或 tuple(Tensor) 的形式。 |

### 转写示例

#### *tensors：一组输入 Tensor

```python
# PyTorch 写法
torch.cartesian_prod(a, b)

# Paddle 写法
paddle.cartesian_prod([a, b])
```
