## [ 仅参数名不一致 ]torch.reshape
### [torch.reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html?highlight=reshape#torch.reshape)

```python
torch.reshape(input,
              shape)
```

### [paddle.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reshape_cn.html#reshape)

```python
paddle.reshape(x,
               shape,
               name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 N-D Tensor ，仅参数名不一致。                   |
| shape        | shape            | 表示输出 Tensor 的 shape 。                   |
