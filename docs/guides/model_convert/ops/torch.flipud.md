## torch.flipup
### [torch.flipud](https://pytorch.org/docs/stable/generated/torch.flipud.html?highlight=flipud#torch.flipud)

```python
torch.flipud(input)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x, axis, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor 。                                     |
| -           | axis            | 表示对输入 Tensor 进行翻转的轴， PyTorch 无此参数， Paddle 需要将其设为 0。               |
### 转写示例
```python
# PyTorch 写法
torch.flipud(x)

# Paddle 写法
paddle.flip(x, 0)
```
