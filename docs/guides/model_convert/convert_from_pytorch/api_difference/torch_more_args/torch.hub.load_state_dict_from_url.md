## [ torch 参数更多 ]torch.hub.load_state_dict_from_url

### [torch.hub.load_state_dict_from_url](https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url)

```python
torch.hub.load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False)
```

### [paddle.hub.load_state_dict_from_url](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/hub/load_state_dict_from_url_cn.html#load-state-dict-from-url)

```python
paddle.hub.load_state_dict_from_url(url, model_dir=None, check_hash=False, file_name=None, map_location=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| url | url |  要下载对象的 URL 地址。 |
| model_dir  | model_dir            | 保存下载文件的目录。 |
| map_location        | map_location       | 指定存储位置的映射方式。  |
| progress       | - | 控制进度条显示，一般对网络训练结果影响不大，可直接删除。|
| check_hash        | check_hash | 是否验证文件的 SHA256 哈希值。|
| file_name | file_name| 下载文件的自定义文件名。 |
| weights_only | - | 是否仅加载权重， 一般对网络训练结果影响不大，可直接删除。 |
