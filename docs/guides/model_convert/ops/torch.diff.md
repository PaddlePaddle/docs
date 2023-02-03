## torch.diff
### [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff)

```python
torch.diff(input,
            n=1,
            dim=-1,
            prepend=None,
            append=None)
```

### [paddle.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diff_cn.html#diff)

```python
paddle.diff(x,
            n=1,
            axis=-1,
            prepend=None,
            append=None,
            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 待计算前向差值的输入 Tensor。                      |
| dim          | axis         | 沿着哪一维度计算前向差值，默认值为-1，也即最后一个维度。 |
