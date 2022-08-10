import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class Metric(object):
	def __init__(self):
		pass

	@abc.abstractmethod
	def reset(self):
		"""
		Reset states and result
		"""
		raise NotImplementedError("function 'reset' not implemented in {}.".format(self.__class__.__name__))

	@abc.abstractmethod
	def update(self, *args):
		"""
		Update states for metric
		Inputs of :code:`update` is the outputs of :code:`Metric.compute`,
		if :code:`compute` is not defined, the inputs of :code:`update`
		will be flatten arguments of **output** of mode and **label** from data:
		:code:`update(output1, output2, ..., label1, label2,...)`
		see :code:`Metric.compute`
		"""
		raise NotImplementedError("function 'update' not implemented in {}.".format(self.__class__.__name__))

	@abc.abstractmethod
	def accumulate(self):
		"""
		Accumulates statistics, computes and returns the metric value
		"""
		raise NotImplementedError("function 'accumulate' not implemented in {}.".format(self.__class__.__name__))

	@abc.abstractmethod
	def name(self):
		"""
		Returns metric name
		"""
		raise NotImplementedError("function 'name' not implemented in {}.".format(self.__class__.__name__))

	def compute(self, *args):
		"""
		This API is advanced usage to accelerate metric calculating, calulations
		from outputs of model to the states which should be updated by Metric can
		be defined here, where Paddle OPs is also supported. Outputs of this API
		will be the inputs of "Metric.update".
		If :code:`compute` is defined, it will be called with **outputs**
		of model and **labels** from data as arguments, all outputs and labels
		will be concatenated and flatten and each filed as a separate argument
		as follows:
		:code:`compute(output1, output2, ..., label1, label2,...)`
		If :code:`compute` is not defined, default behaviour is to pass
		input to output, so output format will be:
		:code:`return output1, output2, ..., label1, label2,...`
		see :code:`Metric.update`
		"""
		return args


class DetectionF1(Metric):
	def __init__(self, pos_label=1, name='DetectionF1', *args, **kwargs):
		super(DetectionF1, self).__init__(*args, **kwargs)
		self.pos_label = pos_label
		self._name = name
		self.reset()

	def update(self, preds, labels, length, *args):
		# [B, T, 2]
		pred_labels = preds.argmax(axis=-1)
		for i, label_length in enumerate(length):
			pred_label = pred_labels[i][1:1 + label_length]
			label = labels[i][1:1 + label_length]
			# the sequence has errors
			if (label == self.pos_label).any():
				if (pred_label == label).all():
					self.tp += 1
				else:
					self.fn += 1
			else:
				if (label != pred_label).any():
					self.fp += 1

	def reset(self):
		"""
		Resets all of the metric state.
		"""
		self.tp = 0
		self.fp = 0
		self.fn = 0

	def accumulate(self):
		precision = np.nan
		if self.tp + self.fp > 0:
			precision = self.tp / (self.tp + self.fp)
		recall = np.nan
		if self.tp + self.fn > 0:
			recall = self.tp / (self.tp + self.fn)
		if self.tp == 0:
			f1 = 0.0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		return f1, precision, recall

	def name(self):
		"""
		Returns name of the metric instance.
		Returns:
			str: The name of the metric instance.
		"""
		return self._name


class CorrectionF1(DetectionF1):
	def __init__(self, pos_label=1, name='CorrectionF1', *args, **kwargs):
		super(CorrectionF1, self).__init__(pos_label, name, *args, **kwargs)

	def update(self, det_preds, det_labels, corr_preds, corr_labels, length, *args):
		# [B, T, 2]
		det_preds_labels = det_preds.argmax(axis=-1)
		corr_preds_labels = corr_preds.argmax(axis=-1)

		for i, label_length in enumerate(length):
			# Ignore [CLS] token, so calculate from position 1.
			det_preds_label = det_preds_labels[i][1:1 + label_length]
			det_label = det_labels[i][1:1 + label_length]
			corr_preds_label = corr_preds_labels[i][1:1 + label_length]
			corr_label = corr_labels[i][1:1 + label_length]

			# The sequence has any errors.
			if (det_label == self.pos_label).any():
				corr_pred_label = corr_preds_label * det_preds_label
				corr_label = det_label * corr_label
				if (corr_pred_label == corr_label).all():
					self.tp += 1
				else:
					self.fn += 1
			else:
				if (det_label != det_preds_label).any():
					self.fp += 1
