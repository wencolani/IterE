import argparse
import logging
import tensorflow as tf
import os

from data import Data
from model import IterE
from experiment import Experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
	parser = argparse.ArgumentParser(description='Experiment setup')
	# misc(short for miscellaneous)
	parser.add_argument('--debug', dest='debug', type=bool, default=True)
	# data property
	parser.add_argument('--datadir', dest='datadir', type=str, default="./datasets/WN18-sparse" )
	# model architecture
	parser.add_argument('--model', dest='model', type=str, default='ANALOGY')
	parser.add_argument('--dim', dest='dim', type=int, default=8)
	parser.add_argument('--loss_type', dest='loss_type', type=str, default='ANALOGY')
	parser.add_argument('--axiom_probability', dest='axiom_probability', type=float, default=0.8)
	# optimization
	parser.add_argument('--lr', dest='lr', type=float, default=0.01)
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=1024)
	parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
	parser.add_argument('--regularizer_type', dest='regularizer_type', type=str, default='L1')
	parser.add_argument('--regularizer_weight', dest='regularizer_weight', type=float, default=1e-3)
	# experiment
	parser.add_argument('--train', dest='train', type=bool, default=True)
	parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=10)
	parser.add_argument('--update_axiom_per', dest='update_axiom_per', type=int, default=2)
	parser.add_argument('--DEBUG', dest='DEBUG', type=bool, default=True)
	parser.add_argument('--neg_samples', dest='neg_samples', type=int, default=1)
	parser.add_argument('--axiom_weight', dest='axiom_weight', type=float, default=0.1)
	parser.add_argument('--device', dest='device', type=str, default='/cpu')
	parser.add_argument('--triple_generator', dest='triple_generator', type=int, default=3)
	parser.add_argument('--axiom_generator', dest='axiom_generator', type=int, default=3)
	parser.add_argument('--save_per', dest='save_per', type=int, default=50)
	parser.add_argument('--save_dir', dest='save_dir', type=str, default='./save/AE')
	parser.add_argument('--init_bound', dest='init_bound', type=float, default=1e-2)
	parser.add_argument('--load_dir', dest='load_dir', type=str, default=None)
	parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=None)
	parser.add_argument('--delay_lr_epoch', dest='delay_lr_epoch', type=int, default=None)
	parser.add_argument('--max_entailment', dest='max_entailment', type=int, default=1000)
	parser.add_argument('--inject_triple_percent', dest='inject_triple_percent', type=float, default=-1.0)

	# evaluation
	parser.add_argument('--num_test', dest='num_test', type=int, default=100)
	parser.add_argument('--test', dest='test', type=bool, default=False)
	parser.add_argument('--test_per_iter', dest='test_per_iter', type=int, default=1)
	parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=10)


	# dest for parser
	option = parser.parse_args()
	print(option)
	if option.DEBUG:
		logging.basicConfig(filename='axiomEmbedding.log', level=logging.DEBUG, filemode='w')
	else:
		logging.basicConfig(filename='axiomEmbedding.log', level=logging.INFO, filemode='w')

	# data pre-process
	data = Data(option.datadir, option.axiom_probability, option.neg_samples,
				axiom_weight=option.axiom_weight,
				max_entialments=option.max_entailment,
				inject_triple_percent = option.inject_triple_percent)
	model = IterE(option, data)

	saver = tf.train.Saver(max_to_keep=None)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = False
	config.log_device_placement = False
	config.allow_soft_placement = True
	config.gpu_options.per_process_gpu_memory_fraction=0.48

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print("Session intilized.")
		# add num_batch to data
		data.reset(option.batch_size)
		experiment = Experiment(sess, option, model, data, saver)

		if option.train:
			experiment.train()

		if option.test:
			experiment.test_only(load_model=option.load_dir)

		for generator in experiment.data_generators:
			generator.terminate()

if __name__ == "__main__":
	main()
