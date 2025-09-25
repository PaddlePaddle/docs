## [ paddle 参数更多 ]torch.nn.parameter.Parameter
### [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter)

```python
torch.nn.parameter.Parameter(data=None,
                             requires_grad=True)
```

### [paddle.base.framework.EagerParamBase.from_tensor](https://github.com/PaddlePaddle/Paddle/blob/31b8fe1ff79d7c0121ab371cba310d1faf7792e8/python/paddle/base/framework.py#L7682)

```python
paddle.base.framework.EagerParamBase.from_tensor(tensor, trainable=True, optimize_attr={'learning_rate': 1.0}, regularizer=None, do_model_average=None, need_clip=True)
```

两者功能一致，但 paddle 参数更多，且第一个参数默认值不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle     | 备注                                                                                        |
| ------------- | ---------------- | -------------------------------------------------------------------------------------------|
| data          | tensor           | 输入 tensor，参数默认值不一致，torch 为可选参数，paddle 为必选参数，当 torch 未输入该参数时需要转写。   |
| requires_grad | trainable        | 是否可训练，仅参数名不一致。                                                                    |
| -             | optimize_attr    | 参数优化的选项。PyTorch 无此参数，Paddle 保持默认即可。                                           |
| -             | regularizer      | 参数正则化类型。PyTorch 无此参数，Paddle 保持默认即可。                                           |
| -             | do_model_average | 是否做模型平均。PyTorch 无此参数，Paddle 保持默认即可。                                           |
| -             | need_clip        | 是否做梯度裁剪。PyTorch 无此参数，Paddle 保持默认即可。                                           |


### 转写示例
#### data 参数：输入 tensor
```python
# torch 可不输入，paddle 必须输入，此时使用 0-D tensor 输入
# PyTorch 写法
torch.nn.parameter.Parameter()

# Paddle 写法
paddle.base.framework.EagerParamBase.from_tensor(paddle.to_tensor([]))
```
