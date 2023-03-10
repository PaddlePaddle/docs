## torch.ravel
### [torch.ravel](https://pytorch.org/docs/stable/generated/torch.ravel.html?highlight=ravel#torch.ravel)

```python
torch.ravel(input)
```

### [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flatten_cn.html)

```python
paddle.flatten(x, start_axis=0, stop_axis=- 1, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor 。                                     |
| -           | start_axis            | 表示 flatten 展开的起始维度， PyTorch 无此参数， Paddle 需要将其设为 0。               |
| -           | stop_axis            | 表示 flatten 展开的结束维度， PyTorch 无此参数， Paddle 需要将其设为-1。               |
### 转写示例
```python
# PyTorch 写法
y = torch.flatten(x)

# Paddle 写法
y = paddle.flatten(x, start_axis=0, stop_axis=-1)
```
