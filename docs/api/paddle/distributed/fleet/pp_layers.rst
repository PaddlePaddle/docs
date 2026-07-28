pp_layers
--------------------

..py:class::paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers


方法
..........
..........
issubclass(layer_func, Layer)
'''''''''

判断layer_func是否为Layer类的封装

**参数**
     - **layer_func** (object) 基于Layer类的封装
     - **Layer** (object) 神经网络层的类，包含一系列说明
**返回**
bool
'''''''''

将神经网络层信息转换成字符串

**参数**
     - **name** (string) layer_func的名称
     - **input** (Tensor) 输入的张量
     - **kwargs** (dict) 解析出的字典参数
**返回**
string

**代码示例**
..code-block::python
            layer_to_str(self.layer_func.__name__, self.inputs,
                           self.kwargs)


sharedLayerDesc类：

方法
......
......
''''''''

共享的神经网络层属性，以类的形式封装

**参数**
     - **key** (int) 以键值方式标明神经网络层的位置
     - **layer_func** (object) 网络层的对象封装
     - **forward_func** (object) 针对共享层的共享参数的封装
     - **shared_weight_attr** (Tensor) 共享的权重参数属性
     - **input** (Tensor) 输入的张量
     - **kwargs** (dict) 解析出的字典参数
**返回**
None

SegmentLayers类：

方法
.......
.......
init(layer_desc, num_parts, method, num_virtual_pipeline_stage)
'''''''
此方法的主要目的用作将一个完整的神经网络层进行分离

**参数**
     - **layer_desc** (object) LayerDesc的对象
     - **num_parts** (int) 神经网络层的分割分数
     - **method** (string) 表示分割方法
     - **num_virtual_pipeline_stages** (int) 流水线并行的层数
**返回**
None

**代码示例**
..code-block::python
         __init__(self,
                 layers_desc,
                 num_parts,
                 method="uniform",
                 num_virtual_pipeline_stage=None)：
           self._layers_desc = layers_desc
           self.method = method
           self.num_parts = num_parts
           self.num_items = len(layers_desc)
           self.num_virtual_pipeline_stage = num_virtual_pipeline_stage

uniform(num_items, num_parts)
''''''''''
此方法是表示将网络层分割的方法

**参数**
      - **num_items** 神经网络层的长度
Tensor


PipelineLayerChunk类：

方法
...........
...........
append(sublayer)
'''''''''''
**参数**
      - **sublayer** (object) 神经网路层的子层
**返回**
None

PipelineLayer类：

方法
............
............
''''''''''''
Pipeline类的初始化

**参数**
    - **layers** (object) 关于流水线并行中神经网络层的结构说明
    - **num_stages** (int) 流水线并行模型的度
    - **topology** (object) 混合并行(张量并行，流水线并行等) GPU拓扑图
    - **loss_fn** (object)  梯度下降法中的损失函数
    - **seg_method** (string) 表示分割流水线并行的网络层的方法名
    - **recompute_interval** (int) 表示每个多少个神经网络层进行重计算
    - **num_virtual_pipeline_stages** (int) 针对被分割后的神经网络层的度
**返回**
None

**代码示例**
..code-block::python
       __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seg_method="uniform",
                 recompute_interval=0,
                 recompute_ctx=None,
                 num_virtual_pipeline_stages=None)

get_stage_from_index(layer_idx)
''''''''''''''
将虚拟分割后的神经网络层与真实的神经网路层进行重定位

**参数**
      - **layer_index** (int) 神经网络层分割并行后的虚拟网络层目录
**返回**
Int

**代码示例**
..code-block::python
        def get_stage_from_index(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers, "layer_idx is out of bound"
        for virtual_pp_rank in range(self._num_virtual_pipeline_stages):
            start_idx = virtual_pp_rank * self._num_virtual_pipeline_stages
            for stage in range(self._num_stages):
                if self.segment_parts[start_idx +
                                      stage] <= layer_idx < self.segment_parts[
                                          start_idx + stage + 1]:
                    return stage

_segment_network_for_interleave(seg_method)
'''''''''''''''
此方法用于将切片后的模型网络层插入至对应的流水线度中

**参数**
        - **seg_method** (string) 表示分割流水线并行的网络层的方法名
**返回**
None

**代码示例**
..code-block::python
       def _segment_network_for_interleave(self, seg_method):
        logger.info("start segment network for interleave scheduler")
        seg = SegmentLayers(
            self._layers_desc,
            num_parts=self._num_stages,
            method=seg_method,
            num_virtual_pipeline_stage=self._num_virtual_pipeline_stages)
        self.segment_parts = seg.do_segment()

        logger.info("segment result:" +
                    ", ".join(str(arg) for arg in self.segment_parts))

        for i in range(self._stage_id, self._total_stages_with_virtual_stages,
                       self._num_virtual_pipeline_stages):
            assert self.segment_parts[i] <= self.segment_parts[i + 1]
            self._start_poss.append(self.segment_parts[i])
            self._end_poss.append(self.segment_parts[i + 1])

        assert len(self._start_poss) == len(self._end_poss)

        self._print_segmentation_for_debug()

_build_layer_impl(satrt, end)
'''''''''''''
将网络层的封装和序号加入至运行函数描述列表中

**参数**
     - **start** (index) 网络层的起始序列号
     - **end** (index)  网络层的终止序列号
**返回**
List

**代码示例**
..code-block::python
      def _build_layer_impl(self, start, end):
        if self._num_virtual_pipeline_stages > 1:
            run_function = PipelineLayerChunk()
        else:
            run_function = self.run_function

        for index, layer in enumerate(self._layers_desc[start:end]):
            layer_index = start + index
            if isinstance(layer, Layer):
                run_function.append(layer)
                if self._num_virtual_pipeline_stages == 1:
                   self.add_sublayer(str(layer_index), layer)
            elif isinstance(layer, SharedLayerDesc):
                if layer.layer_name not in self.shared_layers:
                    self.shared_layers[layer.layer_name] = layer.build_layer()
                    self.shared_weight_attrs[
                        layer.layer_name] = layer.shared_weight_attr
                    for param in self.shared_layers[
                            layer.layer_name].parameters():
                        setattr(param, "is_firstly_shared", True)

                if layer.forward_func is None:
                    run_function.append(self.shared_layers[layer.layer_name])

                else:
                    run_function.append(
                        partial(layer.forward_func,
                                self.shared_layers[layer.layer_name]))

            elif isinstance(layer, LayerDesc):
                model = layer.build_layer()
                run_function.append(model)
                if self._num_virtual_pipeline_stages == 1:
                     self.add_sublayer(str(layer_index), model)
            else:
                run_function.append(layer)

        return run_function

