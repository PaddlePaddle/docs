## [ 仅 API 调用方式不一致 ]torch.nn.Module.train

### [torch.nn.Module.train](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train)

```python
torch.nn.Module.train(mode=True)
```

### [paddle.nn.Layer.train](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/train_cn.html#paddle/nn/Layer/train_cn#cn-api-paddle-nn-Layer-train)

```python
paddle.nn.Layer.train()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class TheModelClass(torch.nn.Module):
    def forward(self, x):
        return x

model = TheModelClass()
model.train()
# Paddle 写法
class TheModelClass(paddle.nn.Layer):
    def forward(self, x):
        return x

model = TheModelClass()
model.train()
```
