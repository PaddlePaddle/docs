## [ torch 参数更多 ] torch.nn.Module.load_state_dict
### [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

```python
torch.nn.Module.load_state_dict(state_dict, strict=True)
```

### [paddle.nn.Layer.set_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#set-state-dict-state-dict-use-structured-name-true)

```python
paddle.nn.Layer.set_state_dict(state_dict, use_structured_name=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| state_dict    | state_dict   | 包含所有参数和可持久性 buffers 的 dict。     |
| strict        | -            | 设置所加载参数字典的 key 是否能够严格匹配，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| -             | use_structured_name | 是否将使用 Layer 的结构性变量名作为 dict 的 key，PyTorch 无此参数，Paddle 保持默认即可。 |
