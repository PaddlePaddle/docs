## [ 仅参数名不一致 ]torch.nn.ParameterList
### [torch.nn.ParameterList](https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html?highlight=nn+parameterlist#torch.nn.ParameterList)

```python
torch.nn.ParameterList(values=None)
```

### [paddle.nn.ParameterList](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ParameterList_cn.html#parameterlist)

```python
paddle.nn.ParameterList(parameters=None)
```
两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| values        | parameters   | 可迭代的 Parameters，参数名不一致。                   |
