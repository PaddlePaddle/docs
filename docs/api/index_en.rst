==================
API Reference
==================

PaddlePaddle (PArallel Distributed Deep LEarning) is an efficient, flexible, and extensible deep learning framework, commits to making the innovation and application of deep learning technology easier.

In this version, PaddlePaddle has made many optimizations to the APIs. You can refer to the following table to understand the API directory structure and description of the latest version of PaddlePaddle.In addition, you can refer to PaddlePaddle's `GitHub <https://github.com/PaddlePaddle/Paddle>`_ for details, or read `Release Notes <../release_note_en.html>`_ to learn about the features of the new version.

**The API directory structure of PaddlePaddle is as follows:**

+-------------------------------+-------------------------------------------------------+
| Directory                     | Functions and Included APIs                           |
+===============================+=======================================================+
| paddle.*                      | The aliases of commonly used APIs are reserved in the |
|                               | paddle root directory, which currently include all    |
|                               | the APIs in the paddle.tensor, paddle.framework and   |
|                               | paddle.device directories                             |
+-------------------------------+-------------------------------------------------------+
| paddle.tensor                 | APIs related to tensor operations such as creating    |
|                               | zeros, matrix operation matmul, transforming concat,  |
|                               | computing add, and finding argmax                     |
+-------------------------------+-------------------------------------------------------+
| paddle.framework              | PaddlePaddle universal APIs and dynamic graph APIs    |
|                               | such as no_grad, save and load.                       |
+-------------------------------+-------------------------------------------------------+
| paddle.device                 | Device management related APIs, such as set_device,   |
|                               | get_device, etc.                                      |
+-------------------------------+-------------------------------------------------------+
| paddle.linalg                 | Linear algebra related APIs, such as det, svd, etc.   |
+-------------------------------+-------------------------------------------------------+
| paddle.fft                    | Fast Fourier Transform related APIs, such as          |
|                               | fft, fft2, etc.                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.amp                    | Paddle automatic mixed precision strategy, including  |
|                               | auto_cast, GradScaler, etc.                           |
+-------------------------------+-------------------------------------------------------+
| paddle.autograd               | Auto grad API, including backward, PyLayer, etc.      |
+-------------------------------+-------------------------------------------------------+
| paddle.callbacks              | Paddle log callback APIs, including ModelCheckpoint,  |
|                               | ProgBarLogger, etc.                                   |
+-------------------------------+-------------------------------------------------------+
| paddle.distributed            | Distributed related basic APIs.                       |
+-------------------------------+-------------------------------------------------------+
| paddle.distributed.fleet      | Distributed related high-level APIs.                  |
+-------------------------------+-------------------------------------------------------+
| paddle.hub                    | Model extension API, including list, load, help, etc. |
+-------------------------------+-------------------------------------------------------+
| paddle.io                     | APIs related to data input and output such as         |
|                               | Dataset, and DataLoader                               |
+-------------------------------+-------------------------------------------------------+
| paddle.jit                    | Dynamic graph to static graph APIs, including         |
|                               | to_static, ProgramTranslator, TracedLayer, etc.       |
+-------------------------------+-------------------------------------------------------+
| paddle.metric                 | APIs related to evaluation computation such as        |
|                               | Accuracy and Auc.                                     |
+-------------------------------+-------------------------------------------------------+
| paddle.nn                     | Networking-related APIs such as Linear, Conv2D,       |
|                               | CrossEntropyLoss, RNN，and ReLU, etc.                 |
+-------------------------------+-------------------------------------------------------+
| paddle.onnx                   | APIs related to convert paddle model to ONNX，such as |
|                               | export                                                |
+-------------------------------+-------------------------------------------------------+
| paddld.optimizer              | APIs related to optimization algorithms such as SGD,  |
|                               | Adagrad, and Adam                                     |
+-------------------------------+-------------------------------------------------------+
| paddle.optimizer.lr           | APIs related to learning rate decay, such as          |
|                               | NoamDecay, StepDecay, PiecewiseDecay, etc.            |
+-------------------------------+-------------------------------------------------------+
| paddle.regularizer            | Regularization APIs, including L1Decay, L2Decay, etc. |
+-------------------------------+-------------------------------------------------------+
| paddle.static                 | Basic framework related APIs under static graph,      |
|                               | such as Variable, Program, Executor, etc.             |
+-------------------------------+-------------------------------------------------------+
| paddle.static.nn              | Special APIs for networking under static graph such   |
|                               | as full connect layer fc and control flow             |
|                               | while_loop/cond                                       |
+-------------------------------+-------------------------------------------------------+
| paddle.text                   | The NLP domain API currently includes data sets       |
|                               | related to the NLP domain, such as Imdb and Movielens.|
+-------------------------------+-------------------------------------------------------+
| paddle.utils                  | Utils APIs, including CppExtension, CUDAExtension.    |
+-------------------------------+-------------------------------------------------------+
| paddle.vision                 | Vision domain APIs such as datasets Cifar10,          |
|                               | data processing ColorJitter, and commonly used models |
|                               | like resnet                                           |
+-------------------------------+-------------------------------------------------------+
| paddle.sparse                 | The Sparse domain API.                                |
+-------------------------------+-------------------------------------------------------+
