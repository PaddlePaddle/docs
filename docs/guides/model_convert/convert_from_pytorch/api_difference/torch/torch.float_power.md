## [输入参数类型不一致]torch.float_power

### [torch.float_power](https://pytorch.org/docs/stable/generated/torch.float_power.html#torch-float-power)

```python
torch.float_power(input, exponent, *, out=None)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/pow_cn.html#pow)

```python
paddle.pow(x, y, name=None)
```

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                         |
| -------- | ------------ | ------------------------------------------------------------ |
| input    | x            | 表示指数计算时的底数，PyTorch 参数 input 可以为 `int` 或 `float` 类型的 Number，也可以是 Tensor，Paddle 参数为 Tensor。其中，在 PyTorch 调用该 API 时，参数 input 和 exponent 只要其中一个需要是 Tensor 的数据类型。 |
| exponent | y            | 表示指数计算时的指数，仅参数名不一致。                       |
| out      | -            | 表示输出的 Tensor ，Paddle 无此函数，需要转写。              |

### 转写示例

#### input：表示指数计算时的底数

```python
# PyTorch 写法
torch.float(number_x, tensor_y)

# Paddle 写法
paddle.pow(paddle.to_tensor(number_x), tensor_y)
```

#### out：指定的输出 Tensor

```python
# PyTorch 写法
torch.float_power(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.pow(x.to('float64'), y.to('float64') if isinstance(y, paddle.Tensor) else y), output)
```
