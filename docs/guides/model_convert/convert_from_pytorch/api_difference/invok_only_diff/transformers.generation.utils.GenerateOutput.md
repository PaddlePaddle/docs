## [ 仅 API 调用方式不一致 ]transformers.generation.utils.GenerateOutput
### [transformers.generation.utils.GenerateOutput](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.generation.utils.GenerateOutput)
```python
transformers.generation.utils.GenerateOutput(last_hidden_state: FloatTensor = None, past_key_values: Optional = None, hidden_states: Optional = None, attentions: Optional = None)
```

### [paddleformers.transformers.model_outputs.BaseModelOutput](https://github.com/PaddlePaddle/PaddleFormers/blob/4bbcdc3405d529cdefee7d4e1caa7dfb13715893/paddleformers/transformers/model_outputs.py#L513)
```python
paddleformers.transformers.model_outputs.BaseModelOutput(last_hidden_state: paddle.Tensor = None, past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None, hidden_states: Optional[Tuple[paddle.Tensor]] = None, attentions: Optional[Tuple[paddle.Tensor]] = None)
```


二者功能一致，参数完全一致，用于封装模型的输出。
