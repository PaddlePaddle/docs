## [ 仅 API 调用方式不一致 ]torch.nn.functional.softmax

### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmax)

```python
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

### [paddle.compat.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/softmax_cn.html#paddle/compat/softmax_cn#cn-api-paddle-compat-softmax)

```python
paddle.compat.softmax(input, dim=None, _stacklevel=3, dtype=None, out=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.nn.functional.softmax(x, -1)

# Paddle 写法
result = paddle.compat.softmax(x, -1)

```
