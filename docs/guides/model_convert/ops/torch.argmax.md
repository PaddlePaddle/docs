## torch.argmax
### [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax)

```python
torch.argmax(input)
```

### [paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax)

```python
paddle.argmax(x,
              axis=None,
              keepdim=False,
              dtype='int64',
              name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的多维 Tensor。                   |
| -             | axis         | 指定对输入 Tensor 进行运算的轴，Pytorch 无，保持默认即可。  |
| -             | keepdim      | 是否在输出 Tensor 中保留减小的维度，Pytorch 无，保持默认即可。  |
| -             | dtype        | 输出 Tensor 的数据类型，Pytorch 无，保持默认即可。  |
