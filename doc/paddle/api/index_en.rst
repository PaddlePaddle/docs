==================
API Reference
==================

PaddlePaddle (PArallel Distributed Deep LEarning) is an efficient, flexible, and extensible deep learning framework.
This page lists the APIs supported by PaddlePaddle 2.0-beta. You can view the information of the APIs here.

In addition, you can refer to PaddlePaddle's `GitHub <https://github.com/PaddlePaddle/Paddle>`_ for details, or read `Release Notes <../release_note_en.html>`_ to learn about the features of the new version.

**The API directory structure of PaddlePaddle 2.0-beta is as follows:**

+-------------------------------+-------------------------------------------------------+
| Directory                     | Functions and Included APIs                           |
+===============================+=======================================================+
| paddle.*                      | The aliases of commonly used APIs are reserved in the |
|                               | paddle root directory, which currently include all    |
|                               | the APIs in the paddle.tensor and paddle.framework    |
|                               | directories                                           |
+-------------------------------+-------------------------------------------------------+
| paddle.tensor                 | APIs related to tensor operations such as creating    |
|                               | zeros, matrix operation matmul, transforming concat,  |
|                               | computing add, and finding argmax                     |
+-------------------------------+-------------------------------------------------------+
| paddle.nn                     | Networking-related APIs such as Linear, Conv2D, loss  |
|                               | function, convolution, LSTM，and activation function  |
+-------------------------------+-------------------------------------------------------+
| paddle.static.nn              | Special APIs for networking under a static graph such |
|                               | as input placeholder data/Input and control flow      |
|                               | while_loop/cond                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.static                 | APIs related to the basic framework under a static    |
|                               | graph such as Variable, Program, and Executor         |
+-------------------------------+-------------------------------------------------------+
| paddle.framework              | Universal APIs and imprerative mode APIs such as      |
|                               | to_variable and prepare_context                       |
+-------------------------------+-------------------------------------------------------+
| paddld.optimizer              | APIs related to optimization algorithms such as SGD,  |
|                               | Adagrad, and Adam                                     |
+-------------------------------+-------------------------------------------------------+
| paddle.optimizer.lr_scheduler | APIs related to learning rate attenuation             |
|                               |                                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.metric                 | APIs related to evaluation index computation such as  |
|                               | accuracy and auc                                      |
+-------------------------------+-------------------------------------------------------+
| paddle.io                     | APIs related to data input and output such as save,   |
|                               | load, Dataset, and DataLoader                         |
+-------------------------------+-------------------------------------------------------+
| paddle.device                 | APIs related to device management such as CPUPlace    |
|                               | and CUDAPlace                                         |
+-------------------------------+-------------------------------------------------------+
| paddle.distributed            | Distributed related basic APIs                        |
|                               |                                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.distributed.fleet      | Distributed related high-level APIs                   |
|                               |                                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.vision                 | Vision domain APIs such as datasets, data processing, |
|                               | and commonly used basic network structures like       |
|                               | resnet                                                |
+-------------------------------+-------------------------------------------------------+
| paddle.text                   | NLP domain APIs such as datasets, data processing,    |
|                               | and commonly used basic network structures like       |
|                               | transformer                                           |
+-------------------------------+-------------------------------------------------------+
