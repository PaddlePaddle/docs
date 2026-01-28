## [ paddle 参数更多 ]torch.Tensor.fill_diagonal_
### [torch.Tensor.fill_diagonal_](https://pytorch.org/docs/stable/generated/torch.Tensor.fill_diagonal_.html?highlight=fill_diagonal_#torch.Tensor.fill_diagonal_)
```python
torch.Tensor.fill_diagonal_(fill_value, wrap=False)
```

### [paddle.Tensor.fill_diagonal_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#fill-diagonal-x-value-offset-0-wrap-false-name-none)
```python
paddle.Tensor.fill_diagonal_(value, offset=0, wrap=False, name=None)
```

两者功能一致且参数用法一致，Paddle 支持 ``fill_value`` 作为 ``value`` 的别名，并额外提供 ``offset`` 和 ``name`` 参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                                                                         |
| ---------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| fill_value | value        | Paddle 支持 ``fill_value`` 作为 ``value`` 的别名。                                                                            |
| -          | offset       | 所选取对角线相对原始主对角线位置的偏移量，正向右上方偏移，负向左下方偏移，默认为 0。PyTorch 无此参数，Paddle 保持默认即可。 |
| wrap       | wrap         | 对于 2 维 Tensor，height>width 时是否循环填充，默认为 False。                                                                |
| -          | name         | 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。                                                            |
