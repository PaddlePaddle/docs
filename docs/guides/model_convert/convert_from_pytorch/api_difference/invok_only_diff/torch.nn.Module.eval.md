## [ 仅 API 调用方式不一致 ]torch.nn.Module.eval

### [torch.nn.Module.eval](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)

```python
torch.nn.Module.eval()
```

### [paddle.nn.Layer.eval](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/eval_cn.html#paddle/nn/Layer/eval_cn#cn-api-paddle-nn-Layer-eval)

```python
paddle.nn.Layer.eval()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.Linear(10, 10)
model.eval()

# Paddle 写法
model = paddle.nn.Linear(10, 10)
model.eval()
```
