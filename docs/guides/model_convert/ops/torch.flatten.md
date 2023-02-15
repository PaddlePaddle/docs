## torch.flatten
### [torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten)

```python
torch.flatten(input,
                start_dim=0,
                end_dim=-1)
```

### [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flatten_cn.html#flatten)

```python
paddle.flatten(x,
                start_axis=0,
                stop_axis=-1,
                name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| start_dim    | start_axis   | flatten 展开的起始维度。            |
| end_dim      | stop_axis    | flatten 展开的结束维度。            |
