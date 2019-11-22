.. _cn_api_fluid_layers_masked_select:

masked_select
-------------------------------

.. py:function:: paddle.fluid.layers.masked_select(input, mask)

该OP将根据mask Tensor的真值选取输入Tensor元素，并返回一个一维Tensor

参数：
          - **input** （Variable）- 输入Tensor，数据类型为float32。
          - **mask** （Variable）- mask Tensor， 数据类型为bool。


返回：根据mask选择后的tensor

返回类型：  Variable


**示例代码**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    mask_shape = [4,1]
    shape = [4,4]
    data = np.random.random(mask_shape).astype("float32")
    input_data = np.random.randint(5,size=shape).astype("float32")
    mask_data = data > 0.5

    # print(input_data)
    # [[0.38972723 0.36218056 0.7892614  0.50122297]
    #  [0.14408113 0.85540855 0.30984417 0.7577004 ]
    #  [0.97263193 0.5248062  0.07655851 0.75549215]
    #  [0.26214206 0.32359877 0.6314582  0.2128865 ]]

    # print(mask_data)
    # [[ True]
    #  [ True]
    #  [False]
    #  [ True]]

    input = fluid.data(name="input",shape=[4,4],dtype="float32")
    mask = fluid.data(name="mask",shape=[4,1],dtype="bool")
    result = fluid.layers.masked_select(input=input, mask=mask)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    start = fluid.default_startup_program()
    main = fluid.default_main_program()
    exe.run(start)
    masked_select_result= exe.run(main, feed={'input':input_data, 'mask':mask_data}, fetch_list=[result])
    # print(masked_select)
    # [0.38972723 0.36218056 0.7892614  0.50122297 0.14408113 0.85540855
    #   0.30984417 0.7577004  0.26214206 0.32359877 0.6314582  0.2128865 ]


