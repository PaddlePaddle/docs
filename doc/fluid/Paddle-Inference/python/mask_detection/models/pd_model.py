import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


class Model:
    def __init__(self,
                 model_file,
                 params_file,
                 use_mkldnn=True,
                 use_gpu=False,
                 device_id=0):
        config = AnalysisConfig(model_file, params_file)
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)
        config.enable_memory_optim()

        if use_gpu:
            print("ENABLE_GPU")
            config.enable_use_gpu(100, device_id)

        if use_mkldnn:
            config.enable_mkldnn()
        self.predictor = create_paddle_predictor(config)

    def run(self, img_list):

        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.reshape(img_list[i].shape)
            input_tensor.copy_from_cpu(img_list[i].copy())

        self.predictor.zero_copy_run()

        results = []
        output_names = self.predictor.get_output_names()

        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_tensor(output_names[i])
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        return results
