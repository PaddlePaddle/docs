## [ 仅参数名不一致 ] torch.nn.Module.load_state_dict
### [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

```python
torch.nn.Module.load_state_dict(state_dict, strict=True)
```

### [paddle.nn.Layer.set_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#set-state-dict-state-dict-use-structured-name-true)

```python
paddle.nn.Layer.set_state_dict(state_dict, use_structured_name=True)
```

两者功能完全一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| state_dict         | state_dict        | 包含所有参数和可持久性 buffers 的 dict 。     |
| strict           | use_structured_name        | 设置所加载参数字典的 key 是否能够严格匹配 。     |

### 转写示例
```python
# Pytorch 写法
model.load_state_dict(x, strict=True)

# Paddle 写法
model.set_state_dict(x, use_structured_name=True))
```
