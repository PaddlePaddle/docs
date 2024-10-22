## [ 组合替代实现 ]torch.hub.load_state_dict_from_url

### [torch.hub.load_state_dict_from_url](https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url)
```python
torch.hub.load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False)
```

在给定的 URL 处加载 Torch 序列化对象

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例

```python
# PyTorch 写法
torch.hub.load_state_dict_from_url(url)

# Paddle 写法
paddle.load(paddle.utils.download.get_weights_path_from_url(url))
```
