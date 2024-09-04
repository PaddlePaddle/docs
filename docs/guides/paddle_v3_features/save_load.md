# Save / Load
## 一、功能概要
基于PIR体系基本结构，结合各方需求，设计一套版本管理完备、兼容性好的save_load体系。
#### 1. Model 层面 （采用json文本存储）
- a. 结合PIR的IR结构，设计简洁的序列化协议，保证正确反序列化的基础上降低存储内容。
- b. 重构底层序列化和反序列化机制，实现PIR类型系统，模型结构增加删除修改灵活扩展时，saveload体系灵活扩展，支持新功能的保存和加载。
- c. 设计良好的版本管理和版本间修改的兼容体系，支持新版本兼容读取旧版本模型进行推理训练的功能。
- d. 推理侧希望保证save后的文件体积大小不劣化、序列化以及反序列化速度不可比protobuf慢
- e. （提高要求）底层的机制或工具收敛， 上层api功能明确，无重复。
#### 2. Parameter 层面
- a. 统一原始Save/Load 接口和内部实现，支持c++ 层参数的存储，扩展至 Python 层
- b. 使用旧版本的序列户格式：Python 层使用pickle序列化工具，C++层使用二进制序列化方法。
- c. （提高要求）在C++实现类似的pickle功能，统一两端序列化协议，使得C++ , Python 层保存的模型和参数可以在C++, Python层直接加载。

## 二. API功能变化
1. 用户使用的 Python 端接口与旧 IR 下保持一致。

    |  API类别  |   3.0 后变化   |  分类   |
    |  :----  | :----  | :----  |
    | `paddle.jit.save`  | 无 |   动转静  
    | `paddle.jit.load`  | 无 |  动转静
    | `paddle.save`  | 无 |  动静统一
    | `paddle.load`  | 无 |  动静统一
    | `paddle.static.save`  | `paddle.static.save_pir` |  静态图
    | `paddle.static.load`  | `paddle.static.load_pir` |  静态图
    | `paddle.static.load_program_state`  | 无  |  动静共用


2. C++ 端接口

    save/load功能的序列化与反序列化功能在C++端实现，暴露出的几个接口可以供C++端调用，同时也通过pybind绑定至python端，供Python API使用。

    * `WriteModule` 通过pybind与 `serialize_pir_program`绑定，可以将指定program序列化为json文件存储至文件系统。
    * `ReadModule` 通过pybind与 `deserialize_pir_program` 绑定，可以将指定json文件反序列化为pir program。
    * `SaveFunction` 和 `SaveCombineFunction` 为参数序列化存储接口，在python端pybind为 `save_func` 和 `save_combine_func`
    * `LoadFunction` 和 `LoadCombineFunction` 为参数序列化存储接口，在python端pybind为 `load_func` 和 `load_combine_func`

    具体函数参数说明见 [interface.h](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/include/interface.h)


## 三. 版本管理支持度，版本兼容方案
版本兼容原则为向后兼容，即新版本支持部分旧版本的推理部署，但旧版本无需支持新版本的推理部署。3.0版本将不再支持1.0版本的推理部署，对于2.0版本则通过program_translator进行转换和支持。
<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/version-update.png" style="zoom:50%"/>
</figure>

以下方案讨论3.0以上版本向后兼容情况：
- 3.0框架加载3.1模型，这种需求本身不合理。抛开合理性，若3.1模型不涉及到新特性，默认支持，否则无法支持。

- 3.1 框架加载3.0模型：3.0以后的3.x版本，将在此次更新的新版版本兼容系统中支持对于旧版本的兼容加载和推理部署。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/version-compat.png" style="zoom:50%"/>
</figure>

## 四.设计思路和实现方案：
### 1.model文件设计方案
**主体设计思路与路线**
<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/architecture.png" style="zoom:50%"/>
</figure>

save_load 体系需要完成PIR的类型系统，模型结构 到 序列化文件的互转功能，其中需要实现类型系统和模型结构到序列化结构的对应规则（及序列化协议），再实现IR结构到序列化结构互转的对应功能。

**关键技术点/子模块设计方案**
#### 1.1 协议定义模块
1. 非program序列化内容

    base_code 描述当前文件的内容，版本。
    ```json
    "base_code" : {"magic" : "PIR", "pirversion": 1}
    ```
2. 类型系统序列化协议内容：
 - dialect
 
   Type，Attrbute, Operation是注册在dialect中的结构，在save时需要将dialect信息保存下来，当前框架可以保证不同dialect的string 名称互斥，因此可以使用string作为保存的基本单位：
    ```cpp
    //有save需求
    paddle::dialect::OperatorDialect  -> "pd_op"
    paddle::dialect::CustomOpDialect  -> "custom_op"
    paddle::dialect::OneDNNOperatorDialect -> "onednn_op"

    pir::ControlFlowDialect -> "cf"
    pir::builtinDialect -> "builtin"

    //kernel 级别的dilect没有save需求
    paddle::dialect::KernelDialect
    paddle::dialect::CustomKernelDialect
    paddle::dialect::OneDNNKernelDialect

    //不确定是否有save需求
    cinn::dialect::OperatorDialect
    cinn::dialect::RuntimeDialect

    pir::shape::ShapeDialect
    ```
 - Type/Attribute

   Attribute/Type分为有值和无值，无值的结构保存使用其class名称即可（但考虑到class的名称需要包含域名，内容多，会采用自定义编码表达），有值的结构需要save的内容是各Attrbute/Type storage中的属性，这些属性是反序列化接口的参数列表。内容可以是基本的组件: 整数浮点数；string；std::vector（数组）；bool；point； 和框架内IR结构Type，Attribute。
   - BuiltinDialectType
   ```cpp
    // 无值Type
    pir::Int8Type{"Id" : 自定义编码}
    pir::BFloat16Type{"Id" :自定义编码}
    pir::Float16Type{"Id" :自定义编码}
    pir::Float32Type{"Id" :自定义编码}
    pir::Float64Type{"Id" :自定义编码}
    pir::Int16Type{"Id" :自定义编码}
    pir::Int32Type{"Id" :自定义编码}
    pir::Int64Type{"Id" :自定义编码}
    pir::BoolType{"Id" :自定义编码}
    pir::IndexType{"Id" :自定义编码}
    pir::Complex64Type{"Id" :自定义编码}
    pir::Complex128Type{"Id" :自定义编码}

    // 有值Type
    从xxxTypeStorage 类中的属性确认该Type包含的内容; 
    pir::DenseTensorType{"Id" :自定义编码,
                         "content" : content_json}
    content_json = {
        Type，
        std::vector<int64_t> => pir::DDim，
        string => pir::DataLayout，
        std::vector<std::vector<size_t>> => pir::LoD，
        size_t
    }

    pir::VectorType{"Id" :自定义编码,
                    "content" : content_json}
    content_json = {
        std::vector<Type>;
        size_t;
    }
   ``` 
   - OperatorDialectType
   ```cpp
    paddle::dialect::DenseTensorType = pir::DenseTensorType 
    paddle::dialect::SelectedRowsType{"Id" :自定义编码,
                                    "content" : content_json}
    content_json = {
        Type
        std::vector<int64_t> => pir::Dim
    }
    paddle::dialect::DenseTensorArrayType{"Id" :自定义编码,
                                          "content" : content_json}
    content_json = {
        Type;
        std::vector<int64_t> => pir::Dim;
        string => pir::DataLayout;
    }
   ```
   - KernelDialectType： kernel相关的type没有save需求
   ```json
    paddle::dialect::AllocatedDenseTensorType
    paddle::dialect::AllocatedSelectedRowsType
    paddle::dialect::AllocatedDenseTensorArrayType
   ```

   - ContolflowDialectType： 控制流相关Type由于属于反向引入的相关类型，暂时没有save需求，有需要可添加。
   ```cpp
    pir::ContainerType // 未注册
    pir::StackType
    pir::InletType
    pir::OutletType
   ```
    - BuiltinDialectAttribute
    ```cpp
    pir::BoolAttribute,{"Id" :自定义编码,
                        "content" : bool}
    pir::FloatAttribute,{"Id" :自定义编码,
                         "content" : float }
    pir::DoubleAttribute,{"Id" :自定义编码,
                          "content" : double}
    pir::Int32Attribute,{"Id" :自定义编码,
                         "content" : int32_t}
    pir::Int64Attribute,{"Id" :自定义编码,
                         "content" : int64_t}
    pir::IndexAttribute,{"Id" :自定义编码,
                         "content" : int64_t}
    pir::ArrayAttribute,{"Id" :自定义编码,
                         "content" : std::vector<Attribte>}
    pir::TypeAttribute,{"Id" :自定义编码,
                        "content" : Type}

    pir::StrAttribute,{"Id" :自定义编码,
                       "content" :  std::string}
    pir::TensorNameAttribute,{"Id" :自定义编码,
                              "content" : std::string}
    pir::PointerAttribute,{"Id" :自定义编码,
                           "content" : void*}

    //没有get函数的attribute,增加get函数，且get的参数要是基本类型，基本类型到内置类型的转换
    //float 到 phi::dtype::complex<float> 需要交给get函数转换。
    pir::Complex64Attribute,{"Id" :自定义编码,
                             "content" :float}
    pir::Complex128Attribute,{"Id" :自定义编码,
                              "content" :double}
    ```
    - OperatorDialectAttribute
    ```cpp
    paddle::dialect::IntArrayAttribute,{"Id" :自定义编码,
                        "content" : std::vector<int64_t>} => phi::IntArray
    paddle::dialect::DataTypeAttribute,{"Id" :自定义编码,
                        "content" : std::string(cppType)} => phi::DataType::
    paddle::dialect::PlaceAttribute,{"Id" :自定义编码,
                        "content" : std::string(allocationType)} => phi::Place::
    paddle::dialect::DataLayoutAttribute{"Id" :自定义编码,
                        "content" : std::string()} => phi::DataLayout::
    
    //Attribute 嵌套Attribute
    paddle::dialect::ScalarAttribute{{"Id" :自定义编码,
                                    "content" : Attribute} => phi::scalar
    ```
    - KernelDialectAttribute
    ```cpp
    paddle::dialect::KernelAttribute
    ```
    - ShapeDialectAttribute
    ```cpp
    pir::shape::SymbolAttribute
    ```
    **> 反序列化方式**

    Type / Attribute 的反序列化有统一的接口处理`parseType()` 和 `parseAttribute()`，识别读入的编码后查表（IrContext提供编码到类的map）得到原始类，递归实现构造内部 Type, 再构造外部 Type。

    有值的 Type / Attribute 的需要提供 `deserialize()` 接口。`deserialize()` 保证传入内容值后能够获得C++对象。

    无值的Type可以直接调用相应的get函数进行恢复。如需要对齐实现，可以增加一个相同内容的 `deserialize()` 接口
    ```cpp
    template <typename... Args>                 \
    static concrete_attribute deserialize(pir::IrContext *ctx, Args... args) {        \
        return pir::AttributeManager::template get<concrete_attribute>(ctx,      \
                                                                    args...);
    }
    ```
3. 模型结构序列化协议内容：
    
- **Operation**

    `Operation` 本身可以使用其唯一的 string name 作为标识。

- **Value/Operand**

    `Value` 和 `Operand` 使用 int64 编号, 起始值为1， `Value` 保存在 op 的 `“OpResults” key` 中， `Operand` 保存在 op 的 `“OpOperands”` 中。

    `Value` 需要保存其编码和其 `Type` 的 key/value 对, `Operand` 只需保存其编码，

    `null_type` 的 `Value` 统一用 0 表示，反序列化时需要使用 `fake_value` 处理;

    `Opresult` 的标识主要用于反序列化时根据标识和 build 后的 `Value` 结果对应关系，确定 `Operand` 的传入值。

- **Attribute**

    可变 `Attribute` 都处理为输入（新 IR 保证）

    其他 `Attribute` 保存 name 和其所属类型，类型中包含具体值的内容。

- **OpResult Attribute**

    `OpResult` 中的一些可选的属性，例如 `persistable`， `stop_gradient` 等，都记录在 `Operation` 的参数 map 中，但它们不是 `create op` 接口需要的参数，因此单独保存，在反序列化的时候 `create` 之后进行设置。
    ```json
    // ...
    "Ops":{
            {"Id":"pd_op.full"
             "OpOperands" : []
             "OpResults" : [{"Id": 1,
                          "Type":{"Id":"pir::DenseTensorType",
                                  "Contents": ["pir::FloatType", [1,1], "NCHW", [[1]], 1]}}
                        ]
             "Attr" : [{"Name": "value",
                    "Type":{"Id": "pir::FloatAttribute",
                            "Contents": 1.0}},
                    {"Name":"shape",
                    "Type":{"Id": "pir::ArrayAttribute",
                            "Contents":[2,3]}}
                    ]
             "OpResultsAttr" : [{"Name":"StopGradient"
                                "Type":{"Id": "pir::ArrayAttribute",
                                    "Contents": [true]}},
                                {"Name":"Persistable"
                                    "Type":{"Id": "pir::ArrayAttribute",
                                        "Contents": [false]}}
                               ]
            }
    }
    ```
    **> 反序列化方式**

    Operation的反序列化调用operation的create函数，以对所有OP采用同一套代码进行构建。
    ```cpp
    inputs;//保存operand 和value的关系
    output_types;// 要求value需要保存type
    attributeMap(string, attribute); // 要求能恢复一个attribute，
    pir::OpInfo // ctx->GetRegisteredOpInfo(op_name);


    static Operation *Create(OperationArgument &&op_argument)
  
    static Operation *Create(const std::vector<pir::Value> &inputs,
                            const AttributeMap &attributes,
                            const std::vector<pir::Type> &output_types,
                            pir::OpInfo op_info,
                            size_t num_regions = 0,
                            const std::vector<Block *> &successors = {});
    ```

 - **Program**

    保存时可选用于推理还是训练，由接口中最后一个参数 `trainable` 控制。
    
    训练情况下： `Op` 中包含 `OpResultAttr` 字段，记录 `Opresult` 的 `stop_gradient` 和 `persistable` 信息。

    `program` 中包含了表达模型结构的单位， `Op`， `Region`， `Block`， `program` 内容的层次结构如下，其中没有内容的字段可以不出现；反序列化时默认为空。

    |  key   | value  |
    |  :----  | :----  |
    | `"program"`  | {`"ModuleOp"`: ...<br>`"ParameterMap"`: ...} |
    | `"ModuleOp"`  | {`"Regions"`:[]} |
    | `"Regions"`  | [{Region0},<br>{Region1}] |
    | `{Region}`  | [`"Id"`: {RegionId_x},<br>`"Blocks"`:[]] |
    | `{Block}`  | {`"Id"`: "BlockId_x",<br>`"BlockArgs"`:[]<br>`"Ops"`:[]} |
    | `"Ops"`  | [{op0},<br>{op1}] |
    | `{Op}`  | {`"Id"`:"xxx",<br>`"OpOperands"`:[],<br>`"OpOperands"`:[],<br>`"Attr"`:[],<br>`"OpResultsAttr"`:[]<br>} |

    列表表达多个相同类型的结构并列关系，例如多个`region`，多个`block`，多个`op`；

    `{}` 表达一个数据结构，例如一个`block`，一个`op`。

    **> 压缩优化**
    
    以上述字符串作为非自定义编码内容的save形式会导致存储文件较大。使用缩短字符串进行空间压缩可以带来良好的效果，如果对存储空间有进一步要求可以考虑对字符串进行字节编码。

    - `dielect name` 压缩：创建 `DialectIdMap` 对 dialect name 进行压缩。

        |  key   | value  |
        |  :----  | :----  |
        | `pir::BuiltinDialect::name()`  | “0” |
        | `paddle::dialect::OperatorDialect::name()`  | “1” |
        | `pir::ControlFlowDialect::name()`  | “2” |
        | `paddle::dialect::CustomOpDialect::name()`  | “3” |
        | `paddle::dialect::DistDialect::name()`  | “4” |

    - `schema key` 压缩：对序列化中出现的 key 标识符进行字母缩写，具体的对应关系在 [schema.h](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/include/schema.h#L4) 中。 
    - `parameter op` 压缩：由于 parameter op 可能在模型中出现多次，且其 Attribute 和 OpresultAttribute 较为固定，因此省略 attribute name， 按照约定顺序存储各属性的值。
        ```cpp
        // attr_name ; type
        // is_distributed; array(bool)
        // is_parameter; array(bool)
        // need_clip; array(bool)
        // parameter_name; string
        // persistable; array(bool)
        // stop_gradient; array(bool)
        // trainable; array(bool)
        ```
    **> 示例**
    
    对于一个简单program：
    ```bash
    {
        (%0) = "builtin.parameter" () {is_distributed:[false],is_parameter:[true],need_clip:[true],parameter_name:"fc_0.b_0",persistable:[true],stop_gradient:[false],trainable:[true]} : () -> builtin.tensor<30xf32>
        (%1) = "builtin.parameter" () {is_distributed:[false],is_parameter:[true],need_clip:[true],parameter_name:"fc_0.w_0",persistable:[true],stop_gradient:[false],trainable:[true]} : () -> builtin.tensor<30x30xf32>
        (%2) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"A",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[-1,30],stop_gradient:[true]} : () -> builtin.tensor<-1x30xf32>
        (%3) = "pd_op.matmul" (%2, %1) {stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<-1x30xf32>, builtin.tensor<30x30xf32>) -> builtin.tensor<-1x30xf32>
        (%4) = "pd_op.add" (%3, %0) {stop_gradient:[false]} : (builtin.tensor<-1x30xf32>, builtin.tensor<30xf32>) -> builtin.tensor<-1x30xf32>
        (%5) = "pd_op.relu" (%4) {stop_gradient:[false]} : (builtin.tensor<-1x30xf32>) -> builtin.tensor<-1x30xf32>
        (%6) = "pd_op.full" () {dtype:(pd_op.DataType)float32,place:(pd_op.Place)Place(cpu),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Double)1} : () -> builtin.tensor<1xf32>
        (%7) = "pd_op.scale" (%5, %6) {bias:(Float)0,bias_after_scale:true,stop_gradient:[false]} : (builtin.tensor<-1x30xf32>, builtin.tensor<1xf32>) -> builtin.tensor<-1x30xf32>
        (%8) = "pd_op.fetch" (%7) {col:(Int32)0,name:"fetch_name_0",persistable:[true],stop_gradient:[false]} : (builtin.tensor<-1x30xf32>) -> builtin.tensor<-1x30xf32>
    }
    ```
    经过序列化存储为的program json格式为：
    ```json
    {"base_code":{"magic":"pir","trainable":true,"version":1},
    "program":{"regions":[{"#":"region_0","blocks":[{"#":"block_0","args":[],"ops":[
    {"#":"p","A":[0,1,1,"fc_0.b_0"],"O":{"%":1,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[30],"NCHW",[],0]}},"OA":[1,0,1]},
    {"#":"p","A":[0,1,1,"fc_0.w_0"],"O":{"%":2,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[30,30],"NCHW",[],0]}},"OA":[1,0,1]},
    {"#":"1.data","A":[{"AT":{"#":"0.a_str","D":"A"},"N":"name"},{"AT":{"#":"1.a_intarray","D":[-1,30]},"N":"shape"},{"AT":{"#":"1.a_dtype","D":"float32"},"N":"dtype"},{"AT":{"#":"1.a_place","D":[0,0,""]},"N":"place"}],"I":[],"O":[{"%":3,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":true}]},"N":"stop_gradient"}]},
    {"#":"1.matmul","A":[{"AT":{"#":"0.a_bool","D":false},"N":"transpose_x"},{"AT":{"#":"0.a_bool","D":false},"N":"transpose_y"}],"I":[{"%":3},{"%":2}],"O":[{"%":4,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":false}]},"N":"stop_gradient"}]},
    {"#":"1.add","A":[],"I":[{"%":4},{"%":1}],"O":[{"%":5,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":false}]},"N":"stop_gradient"}]},
    {"#":"1.relu","A":[],"I":[{"%":5}],"O":[{"%":6,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":false}]},"N":"stop_gradient"}]},
    {"#":"1.full","A":[{"AT":{"#":"1.a_intarray","D":[1]},"N":"shape"},{"AT":{"#":"0.a_f32","D":1.0},"N":"value"},{"AT":{"#":"1.a_dtype","D":"float32"},"N":"dtype"},{"AT":{"#":"1.a_place","D":[1,0,""]},"N":"place"}],"I":[],"O":[{"%":7,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[1],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":true}]},"N":"stop_gradient"}]},
    {"#":"1.scale","A":[{"AT":{"#":"0.a_f32","D":0.0},"N":"bias"},{"AT":{"#":"0.a_bool","D":true},"N":"bias_after_scale"}],"I":[{"%":6},{"%":7}],"O":[{"%":8,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":false}]},"N":"stop_gradient"}]},
    {"#":"1.fetch","A":[{"AT":{"#":"0.a_str","D":"fetch_name_0"},"N":"name"},{"AT":{"#":"0.a_i32","D":0},"N":"col"}],"I":[{"%":8}],"O":[{"%":9,"TT":{"#":"0.t_dtensor","D":[{"#":"0.t_f32"},[-1,30],"NCHW",[],0]}}],"OA":[{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":true}]},"N":"persistable"},{"AT":{"#":"0.a_array","D":[{"#":"0.a_bool","D":false}]},"N":"stop_gradient"}]}]}]}]}}
    ```
 **> 更多示例**

 为了方便理解，以下采用未压缩的字符串展示 program 的序列化结果：
- if op：
    ```bash
    {
    (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"i",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[1],stop_gradient:[true]} : () -> pd_op.tensor<1xf32>
        (%1) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"a",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[1],stop_gradient:[true]} : () -> pd_op.tensor<1xf32>
        (%2) = "pd_op.full" () {dtype:(pd_op.DataType)float32,place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[],stop_gradient:[true],value:(Float)3} : () -> pd_op.tensor<f32>
        (%3) = "pd_op.greater_equal" (%0, %2) {stop_gradient:[true]} : (pd_op.tensor<1xf32>, pd_op.tensor<f32>) -> pd_op.tensor<1xb>
        (%4) = pd_op.if (%3) {stop_gradient:[true]} -> pd_op.tensor<1xf32>
        {
            (%5) = "pd_op.add" (%1, %1) {stop_gradient:[true]} : (pd_op.tensor<1xf32>, pd_op.tensor<1xf32>) -> pd_op.tensor<1xf32>
            () = "cf.yield" (%5) {} : (pd_op.tensor<1xf32>) -> 
        } else {
            (%6) = "pd_op.subtract" (%1, %1) {stop_gradient:[true]} : (pd_op.tensor<1xf32>, pd_op.tensor<1xf32>) -> pd_op.tensor<1xf32>
            () = "cf.yield" (%6) {} : (pd_op.tensor<1xf32>) -> 
        }
        (%7) = "pd_op.mean" (%4) {axis:(pd_op.IntArray)[],keepdim:false,stop_gradient:[true]} : (pd_op.tensor<1xf32>) -> pd_op.tensor<f32>
    }

    ```
    if op 的特殊之处在于， if op 中含有多个 `region`，每个 `region` 中有一个 `block`， 因此 `op` 的序列化内容不同，需要特殊处理。
    ```json
    // ...
    {          
        "Id": "pd_op.if",
        "OpOperands":[4],
        "OpResults":[5],
        "Regions":[{"Id": "RegionId_2",
                    "Blocks": [{"Id":"BlockId_2",
                                "BlockArgs":[],
                                "Ops":[
                                        { "Id":"pd_op.add",
                                            "OpOperands":[2,2],
                                            "OpResults":[{"Id":6, 
                                                        "Type":{"Id":"pir::DenseTensorType" 
                                                                "Contents":["pir::FloatType", [1], "NCHW", [[1]], 1]
                                                                },
                                                        }],
                                            "Attr":[],
                                            "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                        },  
                                        { "Id":"cf.yield",                                                                   
                                            "OpOperands":[6],
                                            "OpResults":[],
                                            "Attr":[],
                                            "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                        }
                                        ]
                                }],
                    },
                    {"ID":"RegionId_3",
                    "Blocks":[{"Id":"BlockId_3",
                                "BlockArgs":[],
                                "Ops":[
                                        {"Id":"pd_op.sub", 
                                            "OpOperands":[2,2]
                                            "OpResults":[{"Id":7,
                                                        "Type":{"Id":"pir::DenseTensorType",
                                                                "Contents":["pir::FloatType", [1], "NCHW", [[1]], 1]
                                                                },
                                                        }],
                                            "Attr":[]
                                            "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                        }  
                                        { "Id":"cf.yield",
                                            "OpOperands":[7],
                                            "OpResults":[],
                                            "Attr":[],
                                            "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                        }
                                        ]
                                }]
                    }],
        }
    ```
- while op：
    ```bash
    (%0) = "pd_op.full" () {dtype:(pd_op.DataType)int64,place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Float)0} : () -> pd_op.tensor<1xi64>
    (%1) = "pd_op.full" () {dtype:(pd_op.DataType)int64,place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Float)1} : () -> pd_op.tensor<1xi64>
    (%2) = "pd_op.full" () {dtype:(pd_op.DataType)int64,place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Float)10} : () -> pd_op.tensor<1xi64>
    (%3) = "pd_op.less_than" (%0, %2) {stop_gradient:[true]} : (pd_op.tensor<1xi64>, pd_op.tensor<1xi64>) -> pd_op.tensor<1xb>
    (%4) = "pd_op.while"(cond=%3, inputs=%0) { 
    ^%arg_0
        (%5) = "pd_op.add" (%arg_0, %1) {stop_gradient:[true]} : (pd_op.tensor<1xi64>, pd_op.tensor<1xi64>) -> pd_op.tensor<1xi64>
        (%6) = "pd_op.less_than" (%5, %2) {stop_gradient:[true]} : (pd_op.tensor<1xi64>, pd_op.tensor<1xi64>) -> pd_op.tensor<1xb>
        () = "cf.yield" (%6, %5) {} : (pd_op.tensor<1xb>, pd_op.tensor<1xi64>) -> 
    }
    ```
    whileOp 的特殊之处在于其 `block` 中含有 `blockArg`，`blockArg` 性质和 `Value` 相同。因此编号也遵循 `Value` 规则，但又需要和 `Op` 产生的 `Value` 有区别，使用 `int64_t` 的负数表达编码。
    ```json
    {   "Id":"pd_op.while",
        "OpOperands":[4,[1]]
        "OpResults":[5:{}]
        "Regions":[
                    {"Id": "RegionId_2",
                        "Blocks" :[{"Id": "BlockId_2",
                                    "BlockArgs":[-1:{}]
                                    "Ops":[
                                            {"Id":"pd_op.add",
                                                "OpOperands":[-1,2],
                                                "OpResults":[{"Id":6,
                                                            "Type":{"Id":"pir::DenseTensorType" 
                                                                    "Contents":["pir::FloatType", [1], "NCHW", [[1]], 1]
                                                                    }
                                                            }],
                                                "Attr":[],
                                                "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                                },  
                                                {"Id":"pd_op.less_than" ,
                                                "OpOperands":[6,3],
                                                "OpResults":[{
                                                                "Id":7,
                                                                "Type":{"Id":"pir::DenseTensorType" ,
                                                                        "Contents":["pir::FloatType", [1], "NCHW", [[1]], 1]
                                                                        }
                                                            }],
                                                "Attr":[],
                                                "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                                },
                                                {"Id":"cf.yield",
                                                "OpOperands":[6,7],
                                                "OpResults":[],
                                                "Attr":[],
                                                "OpResultsAttr":["StopGradient":[true], "Persistable":[false]]
                                                }
                                        ]
                                                        
                                    }]
                    }
                    ]

    }
    ```

#### 1.2 序列化模块/反序列化模块
JSon字符串具体如何与基本数据类型进行转换。选择 `nlohmann` 库。该库提供了众多的函数进行便捷的转换操作。[nlohmann json](https://github.com/nlohmann/json/blob/87cda1d6646592ac5866dc703c8e1839046a6806/README.md)

对于 `Op`， `Attr`， `Type` 的序列化和反序列化，在 `IR` 结构的定义中需要提供用于序列化的接口 `name()`; 和用于反序列化的接口 `get()`;

序列化和反序列化模块提供 `serializeToJson()`； `deserializeFromJson()`的模版接口，提供特化的 `IR` 结构和 `Json` 对象的互转功能，这两个接口分别调用 `name()`， `data()`， `get()` 等接口完成。
```cpp
  template<typename T>
    Json SerializeTypeToJson(const T type){
        Json j;
        j["id"] = type.name();
        return j;
    }
    
    template<>
    Json SerializeTypeToJson<pir::VectorType>(const pir::VectorType type){
        Json j;
        j["id"] = type.name();
        Json content = Json::array();
        for (auto type_ : type.data()){
            content.push_back(writeType(type_));
        }
        j["data"] = content;
        return j;
    }
```

`ModuleWriter` / `ModuleReader` 类，承担了读写 `IR` 结构的管理功能，依托于第三方库完成基本单位的读写。
<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/module.png" style="zoom:50%"/>
</figure>

- **ModuleWriter**

    - **功能**：将传入的Program转化为json对象;

    - **属性**：

        `ModuleWriterConfig` 中保存了当前的PirVersion版本；后续可以扩展一些轻量化的全局信息

        `program_` 是要序列化的program;

        `program_json` 是序列化后的json对象；

        `SerializeAttrMap` 是attr类型和序列化字符串的对应关系；

        `SerializeTypeMap` 是type类型和序列化字符串的对应关系；

        `value_id_map` 是序列化的value 和其数字标识的对应关系，用于序列化operands

        `xxx_id_` 是序列化过程中用于排序计数的标识

    - **接口**：

        `writeProgam(...)` 递归调用 `writeRegion(...)`， `writeBlock()`， `writeOp()` ... 完成 program 的 map 写，`writeOp()` 会有特殊的分支，来处理控制流的写。

        `writeOp()` 递归调用 `writeValue()`，`writeOpOperands()`，`writeAttrbute()`， `writeType()` 完成。

- **ModuleReader**

    - **功能**：将传入的json对象恢复为Program;

    - **属性**: 

        `ModuleReaderConfig` 中保存了当前的 PirVersion 版本；后续可以扩展一些轻量化的全局信息.

        `program_json` 要反序列化的 json 对象；

        `program_` 反序列化后的 program；

        `DeSerializeAttrMap` 是序列化字符串和 attr 类型的对应关系； 

        `DeSerializeTypeMap` 是序列化字符串和 type 类型的对应关系；

        `id_value_map` 是反序列化的 value 和其数字标识的对应关系，用于创建 operands。

    - **接口**：

        `GetProgram()` 函数承担 json 解析功能，解析 base_code 中 PirVersion，判断是否与当前版本匹配，不匹配需要触发版本兼容工作，Program 借助 `ReadProgram()`完成

        `ReadProgram()` 递归调用 `ReadRegion(... )`，`ReadBlock()`，`ReadOp()` 完成 program 的 map 解析和 Op 的重建，`ReadOp()` 会有特殊的分支，来处理控制流的读。这些模块后续承担了版本兼容模块的查表修改功能的接入。

        `ReadOp` 递归调用 `ReadValue()`， `ReadOperands()`， `ReadAttrbute / ReadType` 用于解析输入输出，属性和类型。

#### 1.3 （二次开发）类型/参数扩展支持
当有新的类型出现时，需要注册自定义编码和类型的对应关系，并实现一个对应的c++结构和相应的转化函数。

1. 缩略名添加：
- 对于新增 `Dialect`，在[schema.cc](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/src/schema.cc#L4)的 `DialectIdMap` 中注册新的dialect。
    ```cpp
    // 在当前已有的map基础上顺序递增
    insert(paddle::dialect::DistDialect::name(), "4");
    ```
- 对于`Type/Attribute`，在对应的属性/类型定义中添加 `name()` 函数获取对应的缩略名称。
    ```cpp
    // For new Attribute
    static std::string name() { return "a_xxx"; }
    // For new Type
    static std::string name() { return "t_xxx"; }
    ```
2. 如果新增了 `Dialect`，且在其中新增了 `Type/Attribute`，则需在[serialize_utils.h](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/include/serialize_utils.h#L4)中添加对应的序列化方法。
    ```cpp
      // Write functions
      static Json WritePaddleDistType(const pir::Type& type);
      static Json WritePaddleDistAttr(const pir::Attribute& attr);
    ```
    并在相应的 `writeType()` 和 `writeAttr()` 添加上述方法的调用。

    同理，需要在[deserialize_utils.h](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/include/deserialize_utils.h#L4)中添加对应的反序列化方法。

    ```cpp
    // Read functions
    static pir::Type ReadPaddleDistType(const std::string type_name,
                                      Json* type_json,
                                      pir::IrContext* ctx);

    static pir::Attribute ReadPaddleDistAttr(const std::string attr_name,
                                           Json* attr_json,
                                           pir::IrContext* ctx);
    
    ```
    并在相应的 `parseType()` 和 `parseAttr()`中添加上述方法的调用。
3. 对上述函数进行实现，需要调用到相应的 `serializeXToJson` 和 `deserializeXFromJson` 模板。
    ```cpp
    // serializeAttrToJson: TensorDistAttribute 
    template <>
    Json serializeAttrToJson<paddle::dialect::TensorDistAttribute>(
        const paddle::dialect::TensorDistAttribute& attr) {}
    ```
    ```cpp
    // deserializeTypeFromJson: DistDenseTensorType
    template <>
    paddle::dialect::DistDenseTensorType
    deserializeTypeFromJsonIncludeParseType<paddle::dialect::DistDenseTensorType>(
        Json* type_json, pir::IrContext* ctx){}
    ```
4. 在[serialize.cc](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/src/ir_serialize.cc#L4)和[deserialize.cc](https://github.com/PaddlePaddle/Paddle/blob/c1d7f52d021817ddca58a962c4ea704da8275e9b/paddle/fluid/pir/serialize_deserialize/src/ir_deserialize.cc#L4) 中添加对应 `Type/Attribute` 的函数调用逻辑。

#### 1.4 版本兼容修改模块
1. `patch yaml` 配置说明： 详见[Readme.md]()

2. `PatchBuilder` 解析 `yaml` 构建 `json patch`

    版本兼容模块实现了类 `PatchBuilder` 用来控制 `patch` 的构建和应用。

- **pir version**

    在c++端序列化反序列化模块设计独立的`pir_version`，用来控制pir的迭代升级。每次发布新的版本，会建立一个表示记录patch信息的`yaml`文件，该文件的文件名顺序递增，即为当前的`pir_version`。版本兼容模块在`ReadModule`中会比对读入文件中记录的版本信息`file_version`与当前系统的`pir_version`，如果相同则不需要启动版本兼容模块，如果不相同则需进行patch构建。

- **version patch链式法则构建**

    获取到的`file_version`与`pir_version`如果不同，则会在patch目录下从`file_version`到`pir_version`进行遍历搜寻，依次建立各版本的patch并进行`patch merge`，构建出最终版的patch信息。

    `PatchBuilder.BuildPatch`函数为核心的patch构建函数，该函数读入yaml文件，解析并构建出`patch json`，并根据各patch的类型分别存储到内存的map中去。提供各类型的apply函数，在反序列化阶段根据各元素的名称进行`patch apply`。

3. 反序列化中查找`patch map`，进行`patch apply`

    在反序列化阶段，依次在各元素`（op、type、attribute）`递归构建之前进行patch搜寻，检测map中是否含有此ID下的patch，若有则调用相应的`patch appy`函数进行json修改。返回得到经过`patch merge`的新json，再进入原本的构建流程。

### 2. param 设计方案
#### 2.1 主体设计思路与路线
**使用旧IR下的存储协议，新加转换模块适配新IR**

从减少开发成本和适配成本的角度来看，Python端和C++端均使用旧IR下的存储协议，新增对应的模块来适配PIR。

* PIR 下 value 没有 name 属性，新增模块将 program 中创建的 parameter value 与 name 进行映射，并在 scope 中找到对应的 variable，获取 tensor 值。
* PIR 下不再迁移旧 IR 下的 save (combine) op、load (combine) op，新增 function 代替对应 op kernel 功能。

#### 2.2 关键技术点/子模块设计方案
**适配新IR的转换模块以及原模块更新**：对接旧IR下已有的接口和存储结构，新增新IR的适配转换逻辑

- paddle.save & paddle.load：Python端直接调用协议库函数进行参数保存加载
    <figure align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/param_py.png" style="zoom:50%"/>
    </figure>

- paddle.save_vars & paddle.load_vars：适配推理侧，调用C++端功能实现C++端的参数读写

    <figure align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/save_load/param_cpp.png" style="zoom:50%"/>
    </figure>