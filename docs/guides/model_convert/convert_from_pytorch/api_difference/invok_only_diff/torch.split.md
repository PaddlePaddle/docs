## [ 仅 API 调用方式不一致 ]torch.split

### [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html)

```python
torch.split(tensor, split_size_or_sections, dim=0)
```

### [paddle.compat.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/split_cn.html#paddle/compat/split_cn#cn-api-paddle-compat-split)

```python
paddle.compat.split(tensor, split_size_or_sections, dim=0)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.split(a, 2)

# Paddle 写法
result = paddle.compat.split(a, 2)

```
