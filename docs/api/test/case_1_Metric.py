import abc


class Metric(metaclass=abc.ABCMeta):
    r"""
    Base class for metric, encapsulates metric logic and APIs
    Usage:

        .. code-block:: text

            m = SomeMetric()
            for prediction, label in ...:
                m.update(prediction, label)
            m.accumulate()

    Advanced usage for :code:`compute`:

    Metric calculation can be accelerated by calculating metric states
    from model outputs and labels by build-in operators not by Python/NumPy
    in :code:`compute`, metric states will be fetched as NumPy array and
    call :code:`update` with states in NumPy format.
    Metric calculated as follows (operations in Model and Metric are
    indicated with curly brackets, while data nodes not):

        .. code-block:: text

                 inputs & labels              || ------------------
                       |                      ||
                    {model}                   ||
                       |                      ||
                outputs & labels              ||
                       |                      ||    tensor data
                {Metric.compute}              ||
                       |                      ||
              metric states(tensor)           ||
                       |                      ||
                {fetch as numpy}              || ------------------
                       |                      ||
              metric states(numpy)            ||    numpy data
                       |                      ||
                {Metric.update}               \/ ------------------

    Examples:

        For :code:`Accuracy` metric, which takes :code:`pred` and :code:`label`
        as inputs, we can calculate the correct prediction matrix between
        :code:`pred` and :code:`label` in :code:`compute`.
        For examples, prediction results contains 10 classes, while :code:`pred`
        shape is [N, 10], :code:`label` shape is [N, 1], N is mini-batch size,
        and we only need to calculate accurary of top-1 and top-5, we could
        calculate the correct prediction matrix of the top-5 scores of the
        prediction of each sample like follows, while the correct prediction
        matrix shape is [N, 5].

          .. code-block:: python
            :name: code-compute-example

              def compute(pred, label):
                  # sort prediction and slice the top-5 scores
                  pred = paddle.argsort(pred, descending=True)[:, :5]
                  # calculate whether the predictions are correct
                  correct = pred == label
                  return paddle.cast(correct, dtype='float32')

        With the :code:`compute`, we split some calculations to OPs (which
        may run on GPU devices, will be faster), and only fetch 1 tensor with
        shape as [N, 5] instead of 2 tensors with shapes as [N, 10] and [N, 1].
        :code:`update` can be define as follows:

          .. code-block:: python
            :name: code-update-example

              def update(self, correct):
                  accs = []
                  for i, k in enumerate(self.topk):
                      num_corrects = correct[:, :k].sum()
                      num_samples = len(correct)
                      accs.append(float(num_corrects) / num_samples)
                      self.total[i] += num_corrects
                      self.count[i] += num_samples
                  return accs
    """
    pass
