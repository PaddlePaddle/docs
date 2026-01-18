## [ 仅 API 调用方式不一致 ]transformers.modeling_outputs.BaseModelOutputWithPast
### [transformers.modeling_outputs.BaseModelOutputWithPast](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast)
```python
transformers.modeling_outputs.BaseModelOutputWithPast(last_hidden_state: FloatTensor = None, past_key_values: Optional = None, hidden_states: Optional = None, attentions: Optional = None)
```

### [paddleformers.transformers.model_outputs.BaseModelOutputWithPast](https://github.com/PaddlePaddle/PaddleFormers/blob/4bbcdc3405d529cdefee7d4e1caa7dfb13715893/paddleformers/transformers/model_outputs.py#L590)
```python
paddleformers.transformers.model_outputs.BaseModelOutputWithPast(last_hidden_state: paddle.Tensor = None, past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None, hidden_states: Optional[Tuple[paddle.Tensor]] = None, attentions: Optional[Tuple[paddle.Tensor]] = None)
```


二者功能一致，参数完全一致，用于封装模型的输出。
