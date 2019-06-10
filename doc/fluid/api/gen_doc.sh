#!/bin/bash
#python gen_doc.py layers --submodules control_flow device io nn ops tensor learning_rate_scheduler detection metric_op #> layers.rst

for module in layers data_feeder dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler recordio_writer backward average profiler
do
  python gen_doc.py ${module}
done

python gen_doc.py ""
