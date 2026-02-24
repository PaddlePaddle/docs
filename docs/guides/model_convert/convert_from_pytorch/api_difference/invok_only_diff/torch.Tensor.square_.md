## [ 仅 API 调用方式不一致 ]torch.Tensor.square_

### [torch.Tensor.square\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.square_.html#torch.Tensor.square_)

```python
torch.Tensor.square_()
```

### [paddle.square\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/square__cn.html#paddle.square_)

```python
paddle.square_(x, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
x.square_()

# Paddle 写法
paddle.square_(x)
```
