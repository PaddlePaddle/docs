## [ 仅 API 调用方式不一致 ]torch.nn.functional.linear

### [torch.nn.functional.linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear)

```python
torch.nn.functional.linear(input, weight, bias=None)
```

### [paddle.compat.nn.functional.linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/nn/functional/linear_cn.html#paddle.compat.nn.functional.linear)
```python
paddle.compat.nn.functional.linear(input, weight, bias=None)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
out = torch.nn.functional.linear(input, weight, bias)

# Paddle 写法
out = paddle.compat.nn.functional.linear(input, weight, bias)
```
