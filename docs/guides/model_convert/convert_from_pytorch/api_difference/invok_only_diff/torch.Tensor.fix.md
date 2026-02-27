## [ 仅 API 调用方式不一致 ]torch.Tensor.fix

### [torch.Tensor.fix](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.fix.html#torch.Tensor.fix)

```python
torch.Tensor.fix()
```

### [paddle.Tensor.trunc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#trunc-name-none)

```python
paddle.Tensor.trunc(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = input.fix()

# Paddle 写法
result = input.trunc()
```
