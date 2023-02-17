## torch.argmin
### [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmin#torch.argmin)

```python
torch.argmin(input,
             dim=None,
             keepdim=False)
```

### [paddle.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmin_cn.html#argmin)

```python
paddle.argmin(x,
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
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。 |
| -             | dtype        | 输出 Tensor 的数据类型，Pytorch 无，保持默认即可。  |
