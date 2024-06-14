## [ 仅参数名不一致 ]torch.flatten
### [torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten)

```python
torch.flatten(input,
              start_dim=0,
              end_dim=-1)
```

### [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flatten_cn.html#flatten)

```python
paddle.flatten(x,
               start_axis=0,
               stop_axis=-1,
               name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> start_dim </font>    | <font color='red'> start_axis </font>  | 表示 flatten 展开的起始维度，仅参数名不一致。            |
| <font color='red'> end_dim </font>      | <font color='red'> stop_axis </font>    | 表示 flatten 展开的结束维度，仅参数名不一致。            |
