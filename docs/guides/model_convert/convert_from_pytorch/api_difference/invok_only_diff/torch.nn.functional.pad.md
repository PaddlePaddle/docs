## [ 仅 API 调用方式不一致 ]torch.nn.functional.pad

### [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad)

```python
torch.nn.functional.pad(input, pad, mode="constant", value=None)
```

### [paddle.compat.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/pad_cn.html#paddle/compat/pad_cn#cn-api-paddle-compat-pad)

```python
paddle.compat.pad(input, pad, mode="constant", value=0.0)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.nn.functional.pad(x, [0, 0, 0, 0, 0, 1, 2, 3], value=1)

# Paddle 写法
result = paddle.compat.pad(x, [0, 0, 0, 0, 0, 1, 2, 3], value=1)

```
