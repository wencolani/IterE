import numpy as np
import tensorflow as tf
import time
import logging
from multiprocessing import JoinableQueue, Queue, Process
import pickle

class Experiment():
	"""
	This class handles all the experiment related with axiomEmbedding
	including training, axiomDetection, testing

	Args:
	    	sess: a Tensorflow session
	    	saver: a Tensorflow saver
	    	option: option all hyperparameters
	    	model: the axiomEmbedding model
	    	data: a Data object that contains all information about dataset
	"""
	def __init__(self, sess, option, model, data, saver):
		self.sess = sess
		self.option = option
		self.model = model
		self.data = data
		self.saver = saver
		self.start = time.time()
		self.epoch = 0
		self.learning_rate = self.option.lr

		# set and init the training data generator
		self.queue_raw_training_data = JoinableQueue()
		self.queue_training_data = Queue()
		self.data_generators = list()
		for i in range(option.triple_generator):
			self.data_generators.append(Process(target=self.data.negative_triple_generator,
									  args=(self.queue_raw_training_data, self.queue_training_data)))
			self.data_generators[-1].start()


	def train(self):

		if self.option.load_dir is not None:
			self.saver.restore(self.sess, self.option.load_dir)
			print('load model from %s' % (self.option.load_dir))
			self.epoch = self.option.load_epoch

		while (self.epoch<= self.option.max_epoch):
			print('\n')
			print('epoch:', self.epoch)
			logging.info('epoch: %d'%(self.epoch))


			if self.epoch == 0:
				ent_embed, rel_embed = self.sess.run([self.model.entity_embeddings, self.model.relation_embeddings])
				print('init ent embedding:', ent_embed[:10])
				print('init relembedding:', rel_embed[:10])

			# train for one epoch
			loss_epoch, loss_epoch_reg = self.one_epoch_train(epoch=self.epoch)
			print('[epoch:%d] --loss:%.4f  --reg loss:%.4f'           %(self.epoch, loss_epoch,  loss_epoch_reg))

			# save model
			if self.epoch % self.option.save_per == 0:
				model_path = self.saver.save(self.sess,
											 self.option.save_dir,
											 global_step=self.epoch)
				print('Model saved at %s' % (model_path))
				
			# test sample triples
			if self.epoch !=0 and (self.epoch%self.option.test_per_iter == 0 or self.epoch == self.option.max_epoch):
				if self.option.num_test != -1:
					test_num = self.option.num_test
					valid_num = self.option.num_test
					train_num = self.option.num_test
				else:
					test_num = self.data.num_test_triples
					valid_num = self.data.num_valid_triples
					train_num = self.data.num_train_triples

				self.test('test', num_test=test_num)

			if self.epoch % self.option.update_axiom_per == 0 and self.epoch !=0:
				# axioms include probability for each axiom in axiom pool
				# order: ref, sym, tran, inver, sub, equi, inferC
				# update_axioms:
				#			1) calculate probability for each axiom in axiom pool with current embeddings
				#			2) update the valid_axioms
				axioms_probability = self.update_axiom()
				self.data.update_train_triples(epoch = self.epoch, update_per= self.option.update_axiom_per)

				logging.info('axiom_probability: %s' % (axioms_probability))

			self.epoch += 1

	def test_only(self, load_model, axiom=False):
		self.saver.restore(self.sess, load_model)
		print('load model from %s' % (self.option.load_dir))
		if axiom:
			axioms_probability = self.update_axiom()
			self.test('test', num_test=self.option.num_test, output_rank=True, axiom=axiom)
		else:
			self.test('test', num_test=self.data.num_test_triples, output_rank=True)


	def test(self, dataset, num_test, output_rank=False, axiom=False):
		# test_triples: [num_test*num_entity*2, 3]
		# for each test triple, replace head and tail entity
		test_batch = round(num_test/self.option.test_batch_size)
		# scores_head(tail): [num_entity, 0]
		scores_head = np.asarray([]).reshape([-1, self.data.num_entity])
		scores_tail = np.asarray([]).reshape([-1, self.data.num_entity])
		scores_org = []

		for batch in range(test_batch):
			print('test %d/%d'%(batch, test_batch), end='\r')
			self.test_triples, self.test_triples_org = self.generate_test_triples_batch(dataset,batch,num_test)
			feed = {self.model.input_test_triples: self.test_triples}
			scores = self.model.run_test(self.sess, feed)
			scores_org +=  list(self.model.run_test(self.sess, {self.model.input_test_triples: self.test_triples_org}))
			# score_reshape: [num_test * 2, num_entity]
			scores_reshape = scores.reshape([-1, self.data.num_entity])
			# score_head(tail): [num_test, num_entity]
			head = scores_reshape[:int(len(scores_reshape)/2), :]
			tail = scores_reshape[int(len(scores_reshape)/2):, :]
			scores_head = np.concatenate((scores_head, head), axis=0)
			scores_tail = np.concatenate((scores_tail, tail), axis=0)

		if not axiom:
			MR, MR_h, MR_t, MRR, MRR_h, MRR_t, \
				H1, H1_h, H1_t, H3, H3_h, H3_t, H10, H10_h, H10_t, \
				FMR, FMR_h, FMR_t, FMRR, FMRR_h, FMRR_t, \
				FH1, FH1_h, FH1_t, FH3, FH3_h, FH3_t, \
				FH10, FH10_h, FH10_t = self.rank_test_score(scores_head, scores_tail, dataset, scores_org, num_test, output_rank=output_rank)
		else:
			MR, MR_h, MR_t, MRR, MRR_h, MRR_t, \
			H1, H1_h, H1_t, H3, H3_h, H3_t, H10, H10_h, H10_t, \
			FMR, FMR_h, FMR_t, FMRR, FMRR_h, FMRR_t, \
			FH1, FH1_h, FH1_t, FH3, FH3_h, FH3_t, \
			FH10, FH10_h, FH10_t = self.rank_test_score_with_axiom(scores_head, scores_tail, dataset, scores_org, num_test,
														output_rank=output_rank)
		print('[%s][epoch:%d] --MR:%.2f	--MR_h: %.2f	--MR_t:%.2f' % (dataset, self.epoch, MR, MR_h, MR_t))
		print('[%s][epoch:%d] --MRR:%.3f	--MRR_h: %.3f	--MRR_t:%.3f' % (dataset, self.epoch, MRR, MRR_h, MRR_t))
		print("[%s][epoch:%d] --H1:%.3f	--H1_h: %.3f 	--H1_t:%.3f" % (dataset, self.epoch, H1, H1_h, H1_t))
		print("[%s][epoch:%d] --H3:%.3f	--H3_h: %.3f 	--H3_t:%.3f" % (dataset, self.epoch, H3, H3_h, H3_t))
		print("[%s][epoch:%d] --H10:%.3f	--H10_h: %.3f 	--H10_t:%.3f" % (dataset, self.epoch, H10, H10_h, H10_t))

		print('[%s][epoch:%d] --FMR:%.2f	--FMR_h: %.2f	--FMR_t:%.2f' % (dataset, self.epoch, FMR, FMR_h, FMR_t))
		print('[%s][epoch:%d] --FMRR:%.3f	--FMRR_h: %.3f	--FMRR_t:%.3f' % (dataset, self.epoch, FMRR, FMRR_h, FMRR_t))
		print("[%s][epoch:%d] --FH1:%.3f	--FH1_h: %.3f 	--FH1_t:%.3f" % (dataset, self.epoch, FH1, FH1_h, FH1_t))
		print("[%s][epoch:%d] --FH3:%.3f	--FH3_h: %.3f 	--FH3_t:%.3f" % (dataset, self.epoch, FH3, FH3_h, FH3_t))
		print("[%s][epoch:%d] --FH10:%.3f	--FH10_h: %.3f 	--FH10_t:%.3f" % (dataset, self.epoch, FH10, FH10_h, FH10_t))


	def rank_test_score(self,score_head, score_tail, dataset, scores_org, num_test, output_rank=False, with_axiom = True):
		# head/tail_score: [num_test, num_entity]
		head_score = score_head.reshape(-1, self.data.num_entity)
		tail_score = score_tail.reshape(-1, self.data.num_entity)
		if dataset == 'valid':
			test_ids = np.asarray(self.data.valid_ids)[: num_test, :]
		elif dataset == 'test':
			test_ids = np.asarray(self.data.test_ids)[:num_test, :]
		elif dataset == 'train':
			test_ids = np.asarray(self.data.train_ids)[:num_test, :]
		else:
			raise NotImplementedError
		head_score_rank_id = np.argsort(head_score, axis=1)
		tail_score_rank_id = np.argsort(tail_score, axis=1)

		rank_h, rank_t, frank_h, frank_t = [[] for i in range(4)]

		num = 0
		for triple, head_rank_id, tail_rank_id, head_s, tail_s in zip(test_ids, head_score_rank_id, tail_score_rank_id, head_score, tail_score):
			num += 1
			print('testing %d/%d'%(num, num_test), end='\r')
			h,r,t = triple

			# rank without axiom
			rank_head = self.data.num_entity - np.where(head_rank_id == h)[0][0]
			rank_tail = self.data.num_entity - np.where(tail_rank_id == t)[0][0]
			rank_head_filter = rank_head
			rank_tail_filter = rank_tail
			for i in range(rank_head - 1):
				if head_rank_id[self.data.num_entity - 1 - i] in self.data.tr_h_all[(t, r)]:
					rank_head_filter -= 1
			for i in range(rank_tail - 1):
				if tail_rank_id[self.data.num_entity - 1 - i] in self.data.hr_t_all[(h, r)]:
					rank_tail_filter -= 1

			rank_h.append(rank_head)
			rank_t.append(rank_tail)
			frank_h.append(rank_head_filter)
			frank_t.append(rank_tail_filter)

		rank_h, rank_t, frank_h, frank_t = map(lambda x: np.asarray(x), [rank_h, rank_t, frank_h, frank_t])

		MR_h, MR_t, FMR_h, FMR_t = map(lambda x: np.mean(x), [rank_h, rank_t, frank_h, frank_t])
		MRR_h, MRR_t, FMRR_h, FMRR_t = map(lambda x: np.mean(1.0/x), [rank_h, rank_t, frank_h, frank_t])
		H1_h, H1_t, FH1_h, FH1_t = map(lambda x: np.mean(np.asarray(x <= 1, dtype=float)),
									   [rank_h, rank_t, frank_h, frank_t])
		H3_h, H3_t, FH3_h, FH3_t = map(lambda x: np.mean(np.asarray(x <= 3, dtype=float)),
									   [rank_h, rank_t, frank_h, frank_t])
		H10_h, H10_t, FH10_h, FH10_t = map(lambda x: np.mean(np.asarray(x <= 10, dtype=float)),
									   [rank_h, rank_t, frank_h, frank_t])
		MR, FMR, MRR, FMRR,  H1, FH1, H3, FH3, H10, FH10 = map(lambda x, y: (x + y) / 2.0,
												   [MR_h, FMR_h, MRR_h, FMRR_h, H1_h, FH1_h, H3_h, FH3_h, H10_h, FH10_h],
												   [MR_t, FMR_t, MRR_t, FMRR_t, H1_t, FH1_t, H3_t, FH3_t, H10_t, FH10_t])
		if output_rank:
			with open('./save_rank/rank_h_noaxiom.pickle', 'wb') as f: pickle.dump(rank_h, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/rank_t_noaxiom.pickle', 'wb') as f: pickle.dump(rank_t, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/frank_h_noaxiom.pickle', 'wb') as f: pickle.dump(frank_h, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/frank_t_noaxiom.pickle', 'wb') as f: pickle.dump(frank_t, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/test_ids_noaxiom.pickle', 'wb') as f: pickle.dump(test_ids, f, pickle.HIGHEST_PROTOCOL)

		return MR, MR_h, MR_t, \
			   MRR, MRR_h, MRR_t, \
			   H1, H1_h, H1_t, \
			   H3, H3_h, H3_t, \
			   H10, H10_h, H10_t, \
			   FMR, FMR_h, FMR_t, \
			   FMRR, FMRR_h, FMRR_t, \
			   FH1, FH1_h, FH1_t, \
			   FH3, FH3_h, FH3_t, \
			   FH10, FH10_h, FH10_t

	def rank_test_score_with_axiom(self,score_head, score_tail, dataset, scores_org, num_test, output_rank=False, with_axiom = True):
		head_score = score_head.reshape(-1, self.data.num_entity)
		tail_score = score_tail.reshape(-1, self.data.num_entity)
		filter_head = 0
		filter_tail = 0

		if dataset == 'valid':
			test_ids = np.asarray(self.data.valid_ids)[: num_test, :]
		elif dataset == 'test':
			test_ids = np.asarray(self.data.test_ids)[:num_test, :]
		elif dataset == 'train':
			test_ids = np.asarray(self.data.train_ids)[:num_test, :]
		else:
			raise NotImplementedError

		rank_h, rank_t, frank_h, frank_t = [[] for i in range(4)]

		num = 0
		for triple, head_s, tail_s in zip(test_ids,head_score, tail_score):
			num += 1
			print('testing %d/%d' % (num, num_test), end='\r')
			h, r, t = triple

			if h in self.data.infered_tr_h[(t,r)]:
				filter_head += 1
				rank_head = 1
				rank_head_filter = 1
			else:
				head_rank_id = np.argsort(head_s)
				rank_head = self.data.num_entity - np.where(head_rank_id == h)[0][0]
				rank_head_filter = rank_head
				for i in range(rank_head - 1):
					if head_rank_id[self.data.num_entity - 1 - i] in self.data.tr_h_all[(t, r)]:
						rank_head_filter -= 1

			if t in self.data.infered_hr_t[(h,r)]:
				filter_tail += 1
				rank_tail = 1
				rank_tail_filter = 1
			else:
				tail_rank_id = np.argsort(tail_s)
				rank_tail = self.data.num_entity - np.where(tail_rank_id == t)[0][0]
				rank_tail_filter = rank_tail
				for i in range(rank_tail - 1):
					if tail_rank_id[self.data.num_entity - 1 - i] in self.data.hr_t_all[(h, r)]:
						rank_tail_filter -= 1

			rank_h.append(rank_head)
			rank_t.append(rank_tail)
			frank_h.append(rank_head_filter)
			frank_t.append(rank_tail_filter)

		print('\n')
		print('filter_head:', filter_head)
		print('filter_tail:', filter_tail)
		rank_h, rank_t, frank_h, frank_t = map(lambda x: np.asarray(x), [rank_h, rank_t, frank_h, frank_t])

		MR_h, MR_t, FMR_h, FMR_t = map(lambda x: np.mean(x), [rank_h, rank_t, frank_h, frank_t])
		MRR_h, MRR_t, FMRR_h, FMRR_t = map(lambda x: np.mean(1.0 / x), [rank_h, rank_t, frank_h, frank_t])
		H1_h, H1_t, FH1_h, FH1_t = map(lambda x: np.mean(np.asarray(x <= 1, dtype=float)),
								   [rank_h, rank_t, frank_h, frank_t])
		H3_h, H3_t, FH3_h, FH3_t = map(lambda x: np.mean(np.asarray(x <= 3, dtype=float)),
								   [rank_h, rank_t, frank_h, frank_t])
		H10_h, H10_t, FH10_h, FH10_t = map(lambda x: np.mean(np.asarray(x <= 10, dtype=float)),
									   [rank_h, rank_t, frank_h, frank_t])
		MR, FMR, MRR, FMRR, H1, FH1, H3, FH3, H10, FH10 = map(lambda x, y: (x + y) / 2.0,
														[MR_h, FMR_h, MRR_h, FMRR_h, H1_h, FH1_h, H3_h, FH3_h, H10_h,
														FH10_h],
														[MR_t, FMR_t, MRR_t, FMRR_t, H1_t, FH1_t, H3_t, FH3_t, H10_t,
														FH10_t])
		if output_rank:
			with open('./save_rank/rank_h.pickle', 'wb') as f: pickle.dump(rank_h, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/rank_t.pickle', 'wb') as f: pickle.dump(rank_t, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/frank_h.pickle', 'wb') as f: pickle.dump(frank_h, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/frank_t.pickle', 'wb') as f: pickle.dump(frank_t, f, pickle.HIGHEST_PROTOCOL)
			with open('./save_rank/test_ids.pickle', 'wb') as f: pickle.dump(test_ids, f, pickle.HIGHEST_PROTOCOL)

		return MR, MR_h, MR_t, \
			MRR, MRR_h, MRR_t, \
		   	H1, H1_h, H1_t, \
		   	H3, H3_h, H3_t, \
		   	H10, H10_h, H10_t, \
		   	FMR, FMR_h, FMR_t, \
		   	FMRR, FMRR_h, FMRR_t, \
		   	FH1, FH1_h, FH1_t, \
		   	FH3, FH3_h, FH3_t, \
		   	FH10, FH10_h, FH10_t




	def check_infered(self, triple, head_score, tail_score):
		h,r,t = triple
		head_score = list(head_score)
		tail_score = list(tail_score)
		head_id = [i for i in range(len(head_score))]
		tail_id = [i for i in range(len(tail_score))]
		assert len(head_id) == self.data.num_entity
		assert len(tail_id) == self.data.num_entity
		infer_id_h, infer_s_h, infer_id_t, infer_s_t = [[] for i in range(4)]
		left_id_h, left_s_h, left_id_t, left_s_t = [[] for i in range(4)]

		for i in range(len(head_score)):
			if i in self.data.infered_tr_h[(t,r)]:
				infer_id_h.append(head_id[i])
				infer_s_h.append(head_score[i])
			else:
				left_id_h.append(head_id[i])
				left_s_h.append(head_score[i])

			if i in self.data.infered_hr_t[(h,r)]:
				infer_id_t.append(head_id[i])
				infer_s_t.append(head_score[i])
			else:
				left_id_t.append(head_id[i])
				left_s_t.append(head_score[i])

		return infer_id_h, infer_s_h, infer_id_t, infer_s_t,\
			   left_id_h, left_s_h, left_id_t, left_s_t




	def rank_score(self, triple, head_scores, tail_scores):
		h,r,t = triple
		head_score_axiom = []
		head_score_left  = []
		tail_score_axiom = []
		tail_score_left = []
		assert len(head_scores) == len(tail_scores)
		for i in range(len(head_scores)):
			# check axiom entailment for head prediction
			if (i, r, t) in self.data.train_inject_triples:
				head_score_axiom.append(head_scores[i])
			else:
				head_score_left.append(head_scores[i])

			# check aixom entailment for tail prediction
			if (h,r,i) in self.data.train_inject_triples:
				tail_score_axiom.append(tail_scores[i])
			else:
				tail_score_left.append(tail_scores[i])
		# sort the score
		head_score_axiom_rank = -np.sort(-np.asarray(head_score_axiom))
		head_score_left_rank = -np.sort(-np.asarray(head_score_left))
		tail_score_axiom_rank = -np.sort(-np.asarray(tail_score_axiom))
		tail_score_left_rank = -np.sort(-np.asarray(tail_score_left))

		head_score_rank = np.concatenate([head_score_axiom_rank, head_score_left_rank], axis=0)
		tail_score_rank = np.concatenate([tail_score_axiom_rank, tail_score_left_rank], axis=0)
		return head_score_rank, tail_score_rank




	def one_epoch_train(self, epoch):
		self.data.reset(self.option.batch_size)
		learning_rate = self.learning_rate
		if self.option.delay_lr_epoch is not None and epoch>self.option.delay_lr_epoch:
			learning_rate = self.learning_rate/10
		print('learning_rate', learning_rate)
		positive_triple_generator = self.data.generate_train_batch()
		# axiom_generator = self.data.generate_axiom_batch()

		# prepare the positive training tripels
		# each dat is a batch of training data
		for dat in positive_triple_generator:
			self.queue_raw_training_data.put(dat)
		print('raw training data is initialized')

		loss_epoch = 0.0
		loss_epoch_reg = 0.0
		for batch in range(self.data.num_batch_train):
			start = time.time()

			positive_ids_labels, negative_ids_labels = self.queue_training_data.get()
			positive = positive_ids_labels[:, :3]
			negative = negative_ids_labels[:, :3]
			positive_labels = np.reshape(positive_ids_labels[:, -1], [-1, 1])
			negative_labels = np.reshape(negative_ids_labels[:, -1], [-1, 1])


			feed = {self.model.pos_triples: positive,
					self.model.neg_triples: negative,
					self.model.pos_labels: positive_labels,
					self.model.neg_labels: negative_labels,
					self.model.learning_rate: learning_rate}
			loss_batch, loss_reg = self.model.run_train_batch(self.sess, feed)
			loss_epoch += loss_batch
			loss_epoch_reg += loss_reg

			if batch % 20 == 0:
				print('batch/num_batch: %d/%d, loss: %.6f, loss_reg: %.6f'%(batch, self.data.num_batch_train, loss_batch, loss_reg), end='\r')

		ent_embed, rel_embed = self.sess.run([self.model.entity_embeddings, self.model.relation_embeddings])

		return loss_epoch, loss_epoch_reg


	def update_axiom(self):
		time_s = time.time()
		axiom_pro = self.model.run_axiom_probability(self.sess, self.data)
		time_e = time.time()
		print('calculate axiom score:', time_e -time_s)
		with open('./save_axiom_prob/axiom_prob.pickle', 'wb') as f: pickle.dump(axiom_pro, f, pickle.HIGHEST_PROTOCOL)
		with open('./save_axiom_prob/axiom_pools.pickle', 'wb') as f: pickle.dump(self.data.axiompool, f, pickle.HIGHEST_PROTOCOL)
		self.data.update_valid_axioms(axiom_pro)
		return self.model.run_axiom_probability(self.sess, self.data)

	def generate_test_triples_batch(self, type, batch, num_test):
		start = min(num_test, batch*self.option.test_batch_size)
		end = min(start+self.option.test_batch_size, num_test)
		if type=='test':
			test_triple_ids = self.data.test_ids[start:end]
		elif type=='valid':
			test_triple_ids = self.data.valid_ids[start:end]
		elif type=='train':
			test_triple_ids = self.data.train_ids[start:end]
		else:
			raise NotImplementedError
		test_triple_replace = self.replace_test_triple(test_triple_ids)
		return test_triple_replace, test_triple_ids


	def replace_test_triple(self, input_triples):
		input_triples = np.asarray(input_triples)
		replace_head_rt = input_triples[:, 1:]
		replace_tail_hr = input_triples[:, :2]
		replace_ids = np.asarray([i for i in range(self.data.num_entity)]*len(input_triples)).reshape([-1, 1])
		replace_head_repeat = np.repeat(replace_head_rt, self.data.num_entity, axis=0)
		replace_tail_repeat = np.repeat(replace_tail_hr, self.data.num_entity, axis=0)
		replace_head = np.concatenate((replace_ids,replace_head_repeat), axis=1)
		replace_tail = np.concatenate((replace_tail_repeat, replace_ids),axis=1)
		replace_test = np.concatenate((replace_head, replace_tail), axis=0)
		return replace_test




