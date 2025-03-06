..  _api_guide_executor_en:

################
Executor
################

:code:`Executor` realizes a simple executor in which all operators will be executed in order. You can run :code:`Executor` in a Python script.

The logic of :code:`Executor` is very simple. It is suggested to thoroughly run the model with :code:`Executor` in debugging phase on one computer and then switch to mode of multiple devices or multiple computers to compute.

:code:`Executor` accepts either a :code:`Place` including :ref:`api_paddle_CPUPlace` and :ref:`api_paddle_CUDAPlace`, or a device descriptor :code:`str` such as :code:`"cpu"` and :code:`"gpu:0"`.

.. code-block:: python

    import paddle
    import numpy as np

    # Enable static graph mode and configure compute device.
    paddle.enable_static()
    place = (
        paddle.CUDAPlace(0)
        if "gpu:0" in paddle.device.get_available_device()
        else paddle.CPUPlace()
    )

    # First create the Executor to run computation graph.
    exe = paddle.static.Executor(place)

    # define input data and labels
    # shape '-1' indicates dynamic batch size and '784' is the feature dimension
    x = paddle.static.data(name="x", shape=[-1, 784], dtype="float32")
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    # define a fully connected layer with an output dimension of 10, and cross entropy loss function
    out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
    loss = paddle.nn.functional.cross_entropy(out, label)

    # Run the startup program once to initialize parameters.
    exe.run(paddle.static.default_startup_program())

    # create a 1x784 matrix filled with 1 as input data, and a label of 0 (1x1 matrix)
    feed_dict = {
        "x": np.ones((1, 784), dtype="float32"),
        "label": np.array([[0]], dtype="int64"),
    }

    # Run the main program using feed_dict as input and get the output of loss and out.
    (loss_value, out_value) = exe.run(
        paddle.static.default_main_program(), feed=feed_dict, fetch_list=[loss, out]
    )
    print(f"loss: {loss_value}\nmodel output: {out_value}")

For the parameter introduction about Executor please refer to :ref:`api_paddle_static_Executor`
