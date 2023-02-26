## torch.clamp
### [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html?highlight=clamp#torch.clamp)

```python
torch.clamp(input,
            min=None,
            max=None,
            *,
            out=None)
```

### [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/clip_cn.html#clip)

```python
paddle.clip(x,
            min=None,
            max=None,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.clamp([-1.7120,  0.1734, -0.0478, -0.0922], min=-0.5, max=0.5, out=y)

# Paddle 写法
y = paddle.clip([-1.7120,  0.1734, -0.0478, -0.0922], min=-0.5, max=0.5)
```
