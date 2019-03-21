import numpy as np
import os
from collections import defaultdict
from scipy.sparse import coo_matrix
import pickle
import logging
import random
import time

class Data():
	def __init__(self, datadir, select_probability, neg_samples,
				 axiom_weight=1.0,
				 max_entialments = 2000,
				 types=10,
				 inject_triple_percent = 1.0):
		self.axiom_types  = types
		self.select_probability = select_probability
		self.neg_samples = neg_samples
		self.sparsity = 0.995
		self.axiom_weight = axiom_weight
		self.max_entialments = max_entialments
		self.inject_triple_percent = inject_triple_percent

		self.entities = set()
		self.relations = set()
		train_dir, test_dir, valid_dir = map(lambda x: os.path.join(datadir, x), ['train.txt', 'test_sparsity_0.995.txt','valid_sparsity_0.995.txt'])
		#train_dir, test_dir, valid_dir = map(lambda x: os.path.join(datadir, x), ['train.txt', 'test.txt', 'valid.txt'])
		ent2id_dir, rel2id_dir = map(lambda x:os.path.join(datadir, x), ['entity2id.txt', 'relation2id.txt'])
		self.axiom_dir = os.path.join(datadir, 'axiom_pool')
		self.reflexive_dir, self.symmetric_dir, self.transitive_dir, \
		self.inverse_dir, self.subproperty_dir, self.equivalent_dir, \
		self.inferencechain1, self.inferencechain2,\
			self.inferencechain3, self.inferencechain4 = map(lambda x: os.path.join(self.axiom_dir, x),
															 ['axiom_reflexive.txt',
															  'axiom_symmetric.txt',
															  'axiom_transitive.txt',
															  'axiom_inverse.txt',
															  'axiom_subProperty.txt',
															  'axiom_equivalent.txt',
															  'axiom_inferenceChain1.txt',
															  'axiom_inferenceChain2.txt',
															  'axiom_inferenceChain3.txt',
															  'axiom_inferenceChain4.txt'])

		# load train/test/valid dataset
		self.train_triples = self._read_triples(train_dir)
		self.test_triples = self._read_triples(test_dir)
		self.valid_triples = self._read_triples(valid_dir)

		# generate mapping between surface name to id
		self.entity2id, self.relation2id = self._read_id(ent2id_file=ent2id_dir,
														 rel2id_file=rel2id_dir)
		print('read entity2id and relation2id')
		self.id2entity = {v:k for k,v in self.entity2id.items()}
		self.id2reltion = {v:k for k,v in self.relation2id.items()}

		# store data with ids
		self.train_ids = self._triple2id(self.train_triples)
		self.test_ids = self._triple2id(self.test_triples)
		self.valid_ids = self._triple2id(self.valid_triples)


		train_labels = np.reshape(np.ones(len(self.train_ids)), [-1, 1])
		self.train_ids_labels = np.concatenate([self.train_ids, train_labels], axis=1)
		self.train_ids_labels_inject = np.reshape([], [-1, 4])

		# generate r_ht, hr_t
		self.r_ht, self.hr_t, self.tr_h, self.hr_t_all, self.tr_h_all \
			= self._generate(self.train_ids, self.valid_ids, self.test_ids)

		# statistic in dataset
		self.num_entity = len(self.entities)
		self.num_relation = len(self.relations)
		self.num_train_triples = len(self.train_ids)
		self.num_test_triples = len(self.test_ids)
		self.num_valid_triples = len(self.valid_ids)

		# generate entity2frequency and entity2sparsity dict
		self.entity2frequency, self.entity2sparsity = self._entity2frequency()


		# read and materialize axioms
		self._read_axioms()
		self._materialize_axioms()
		self._init_valid_axioms()

	def _entity2frequency(self):
		ent2freq = {ent:0 for ent in range(self.num_entity)}
		ent2sparsity = {ent:-1 for ent in range(self.num_entity)}
		for h,r,t in self.train_ids:
			ent2freq[h] += 1
			ent2freq[t] += 1
		ent_freq_list = np.asarray([ent2freq[ent] for ent in range(self.num_entity)])
		ent_freq_list_sort = np.argsort(ent_freq_list)

		max_freq = max(list(ent2freq))
		min_freq = min(list(ent2freq))
		for ent, freq in ent2freq.items():
			sparsity = 1 - (freq-min_freq)/(max_freq - min_freq)
			ent2sparsity[ent] = sparsity
		return ent2freq, ent2sparsity


	def _read_triples(self, triple_files):
		f = open(triple_files, 'r')
		triples = []
		for line in f.readlines():
			h, t, r = line.strip().split('\t')
			#h, r, t = line.strip().split('\t')
			self.entities.add(h)
			self.entities.add(t)
			self.relations.add(r)
			triples.append((h, r, t))
		return triples

	def _list2id(self, input_list):
		output_dic = dict()
		i=0
		for item in input_list:
			output_dic[item] = i
			i += 1
		return output_dic

	def _read_id(self, ent2id_file, rel2id_file):
		ent2id_f = open(ent2id_file, 'r')
		rel2id_f = open(rel2id_file, 'r')
		ent2id = dict()
		rel2id = dict()
		for line in ent2id_f.readlines():
			line_list = line.strip().split('\t')
			assert  len(line_list) == 2
			ent2id[line_list[0]] = int(line_list[1])

		for line in rel2id_f.readlines():
			line_list = line.strip().split('\t')
			assert len(line_list) == 2
			rel2id[line_list[0]] = int(line_list[1])
		return ent2id, rel2id


	def _triple2id(self, triples):
		output = [(self.entity2id[h], self.relation2id[r], self.entity2id[t])
				  for (h,r,t) in triples]
		return output


	def _generate(self, train, valid, test):
		r_ht = defaultdict(set)
		hr_t = defaultdict(set)
		tr_h = defaultdict(set)
		hr_t_all = defaultdict(list)
		tr_h_all = defaultdict(list)
		for (h,r,t) in train:
			r_ht[r].add((h,t))
			hr_t[(h,r)].add(t)
			tr_h[(t,r)].add(h)
			hr_t_all[(h,r)].append(t)
			tr_h_all[(t,r)].append(h)
		for (h,r,t) in test+valid:
			hr_t_all[(h,r)].append(t)
			tr_h_all[(t, r)].append(h)
		return r_ht, hr_t, tr_h, hr_t_all, tr_h_all

	def _filter_test(self, dataset):
		row_hr_t = []
		col_hr_t = []
		row_tr_h = []
		col_tr_h = []
		num = 0
		for (h,r,t) in dataset:
			# this hr_t includes t and tr_h includes h
			hr_t = self.hr_t_all[(h,r)]
			tr_h = self.tr_h_all[(t,r)]

			# remove the t in hr_t and h in tr_h
			while t in hr_t:
				hr_t.remove(t)
			while h in tr_h:
				tr_h.remove(h)

			col_hr_t += hr_t
			col_tr_h += tr_h
			row_hr_t += [num for i in range(len(hr_t))]
			row_tr_h += [num for i in range(len(tr_h))]
			num += 1
		assert len(row_hr_t) == len(col_hr_t)
		assert len(row_tr_h) == len(col_tr_h)
		data_hr_t = [1.0 for i in range(len(row_hr_t))]
		data_tr_h = [1.0 for i in range(len(row_tr_h))]
		filter_hr_t = coo_matrix((data_hr_t, (row_hr_t, col_hr_t)), shape=(len(dataset), self.num_entity))
		filter_tr_h = coo_matrix((data_tr_h, (row_tr_h, col_tr_h)), shape=(len(dataset), self.num_entity))
		return filter_hr_t, filter_tr_h






	def _read_axioms(self):
		# for each axiom, the first id is the basic relation
		self.axiompool_reflexive = self._read_axiompool_file(self.reflexive_dir)
		self.axiompool_symmetric = self._read_axiompool_file(self.symmetric_dir)
		self.axiompool_transitive = self._read_axiompool_file(self.transitive_dir)
		self.axiompool_inverse = self._read_axiompool_file(self.inverse_dir)
		self.axiompool_equivalent = self._read_axiompool_file(self.equivalent_dir)
		self.axiompool_subproperty = self._read_axiompool_file(self.subproperty_dir)
		self.axiompool_inferencechain1 = self._read_axiompool_file(self.inferencechain1)
		self.axiompool_inferencechain2 = self._read_axiompool_file(self.inferencechain2)
		self.axiompool_inferencechain3 = self._read_axiompool_file(self.inferencechain3)
		self.axiompool_inferencechain4 = self._read_axiompool_file(self.inferencechain4)
		self.axiompool = [self.axiompool_reflexive, self.axiompool_symmetric, self.axiompool_transitive,
						  self.axiompool_inverse, self.axiompool_subproperty, self.axiompool_equivalent,
						  self.axiompool_inferencechain1,self.axiompool_inferencechain2,
						  self.axiompool_inferencechain3,self.axiompool_inferencechain4]


	def _read_axiompool_file(self, file):
		f = open(file, 'r')
		axioms = []
		for line in f.readlines():
			line_list = line.strip().split('\t')
			axiom_ids = list(map(lambda x: self.relation2id[x], line_list))
			#axiom_ids = self.relation2id[line_list]
			axioms.append(axiom_ids)
		# for the case reflexive pool is empty
		if len(axioms) == 0:
			np.reshape(axioms, [-1, 3])
		return axioms

	# for each axioms in axiom pool
	# generate a series of entailments for each axiom
	def _materialize_axioms(self, generate=True, dump=True, load=False):
		if generate:
			self.reflexive2entailment = defaultdict(list)
			self.symmetric2entailment = defaultdict(list)
			self.transitive2entailment = defaultdict(list)
			self.inverse2entailment = defaultdict(list)
			self.equivalent2entailment = defaultdict(list)
			self.subproperty2entailment = defaultdict(list)
			self.inferencechain12entailment = defaultdict(list)
			self.inferencechain22entailment = defaultdict(list)
			self.inferencechain32entailment = defaultdict(list)
			self.inferencechain42entailment = defaultdict(list)

			
			self.reflexive_entailments, self.reflexive_entailments_num = self._materialize_sparse(self.axiompool_reflexive, type='reflexive')
			self.symmetric_entailments, self.symmetric_entailments_num = self._materialize_sparse(self.axiompool_symmetric, type='symmetric')
			self.transitive_entailments, self.transitive_entailments_num = self._materialize_sparse(self.axiompool_transitive, type='transitive')
			self.inverse_entailments, self.inverse_entailments_num = self._materialize_sparse(self.axiompool_inverse, type='inverse')
			self.subproperty_entailments, self.subproperty_entailments_num = self._materialize_sparse(self.axiompool_subproperty, type='subproperty')
			self.equivalent_entailments, self.equivalent_entailments_num  = self._materialize_sparse(self.axiompool_equivalent, type='equivalent')

			self.inferencechain1_entailments, self.inferencechain1_entailments_num = self._materialize_sparse(
				self.axiompool_inferencechain1, type='inferencechain1')
			self.inferencechain2_entailments, self.inferencechain2_entailments_num = self._materialize_sparse(
				self.axiompool_inferencechain2, type='inferencechain2')
			self.inferencechain3_entailments, self.inferencechain3_entailments_num = self._materialize_sparse(
				self.axiompool_inferencechain3, type='inferencechain3')
			self.inferencechain4_entailments, self.inferencechain4_entailments_num = self._materialize_sparse(
				self.axiompool_inferencechain4, type='inferencechain4')


			print('reflexive entailments for sparse: ', self.reflexive_entailments_num)
			print('symmetric entailments for sparse: ', self.symmetric_entailments_num)
			print('transitive entailments for sparse: ', self.transitive_entailments_num)
			print('inverse entailments for sparse: ', self.inverse_entailments_num)
			print('subproperty entailments for sparse: ', self.subproperty_entailments_num)
			print('equivalent entailments for sparse: ', self.equivalent_entailments_num)
			print('inferencechain1 entailments for sparse: ', self.inferencechain1_entailments_num)
			print('inferencechain2 entailments for sparse: ', self.inferencechain2_entailments_num)
			print('inferencechain3 entailments for sparse: ', self.inferencechain3_entailments_num)
			print('inferencechain4 entailments for sparse: ', self.inferencechain4_entailments_num)


			logging.info("finish generate axioms entailments for sparse")


		if dump:
			pickle.dump(self.reflexive_entailments, open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'wb'))
			pickle.dump(self.symmetric_entailments, open(os.path.join(self.axiom_dir, 'symmetric_entailments'), 'wb'))
			pickle.dump(self.transitive_entailments, open(os.path.join(self.axiom_dir, 'transitive_entailments'), 'wb'))
			pickle.dump(self.inverse_entailments, open(os.path.join(self.axiom_dir, 'inverse_entailments'), 'wb'))
			pickle.dump(self.subproperty_entailments, open(os.path.join(self.axiom_dir, 'subproperty_entailments'), 'wb'))
			#pickle.dump(self.inferencechain_entailments, open(os.path.join(self.axiom_dir, 'inferencechain_entailments'), 'wb'))
			pickle.dump(self.equivalent_entailments, open(os.path.join(self.axiom_dir, 'equivalent_entailments'), 'wb'))

			pickle.dump(self.inferencechain1_entailments,
						open(os.path.join(self.axiom_dir, 'inferencechain1_entailments'), 'wb'))
			pickle.dump(self.inferencechain2_entailments,
						open(os.path.join(self.axiom_dir, 'inferencechain2_entailments'), 'wb'))
			pickle.dump(self.inferencechain3_entailments,
						open(os.path.join(self.axiom_dir, 'inferencechain3_entailments'), 'wb'))
			pickle.dump(self.inferencechain4_entailments,
						open(os.path.join(self.axiom_dir, 'inferencechain4_entailments'), 'wb'))

			logging.info("finish dump axioms entialments")

		if load:
			logging.debug("load refexive entailments...")
			self.reflexive_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'rb'))
			logging.debug('load symmetric entailments...')
			self.symmetric_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'symmetric_entailments'), 'rb'))
			logging.debug("load transitive entialments... ")
			self.transitive_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'transitive_entailments'), 'rb'))
			logging.debug("load inverse entailments...")
			self.inverse_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'inverse_entailments'), 'rb'))
			logging.debug("load subproperty entailments...")
			self.subproperty_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'subproperty_entailments'), 'rb'))
			logging.debug("load inferencechain entailments...")
			self.inferencechain_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'inferencechain_entailments'), 'rb'))
			logging.debug("load equivalent entialments...")
			self.equivalent_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'equivalent_entailments'), 'rb'))

			logging.debug("load inferencechain1 entailments...")
			self.inferencechain1_entailments = pickle.load(
				open(os.path.join(self.axiom_dir, 'inferencechain1_entailments'), 'rb'))
			logging.debug("load inferencechain2 entailments...")
			self.inferencechain2_entailments = pickle.load(
				open(os.path.join(self.axiom_dir, 'inferencechain2_entailments'), 'rb'))
			logging.debug("load inferencechain3 entailments...")
			self.inferencechain3_entailments = pickle.load(
				open(os.path.join(self.axiom_dir, 'inferencechain3_entailments'), 'rb'))
			logging.debug("load inferencechain4 entailments...")
			self.inferencechain4_entailments = pickle.load(
				open(os.path.join(self.axiom_dir, 'inferencechain4_entailments'), 'rb'))

			logging.info("finish load axioms entailments")


	def _materialize_sparse(self, axioms, type=None, sparse = False):
		inference = []
		# axiom2entailment is a dict
		# with the all axioms in the axiom pool as keys
		# and all the entailments for each axiom as values
		axiom_list = axioms
		length = len(axioms)
		max_entailments = self.max_entialments
		num = 0
		if length == 0:
			if type == 'reflexive':
				np.reshape(inference, [-1, 3])
			elif type == 'symmetric' or type =='inverse' or  type =='equivalent' or type =='subproperty':
				np.reshape(inference, [-1, 6])
			elif type=='transitive' or type=='inferencechain':
				np.reshape(inference, [-1, 9])
			else:
				raise NotImplementedError
			return inference, num

		if type == 'reflexive':
			for axiom in axiom_list:
				axiom_key =tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h,t) in self.r_ht[r]:
					# filter the axiom with too much entailments
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 2])
						break
					if h != t and self.entity2sparsity[h]>self.sparsity:
						num += 1
						inference_tmp.append([h,r,h])

				for entailment in inference_tmp:
					self.reflexive2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'symmetric':
			#self.symmetric2entailment = defaultdict(list)
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h,t) in self.r_ht[r]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 2])
						break
					if (t,h) not in self.r_ht[r] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
						num += 1
						inference_tmp.append([h,r,t,t,r,h])


				for entailment in inference_tmp:
					self.symmetric2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'transitive':
			#self.transitive2entailment = defaultdict(list)
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h,t) in self.r_ht[r]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					# (t,r,e) exist but (h,r,e) not exist and e!=h
					for e in self.hr_t[(t,r)]- self.hr_t[(h,r)]:
						if e != h and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[e]>self.sparsity):
							num += 1
							inference_tmp.append([h,r,t,t,r,e,h,r,e])

				for entailment in inference_tmp:
					self.transitive2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inverse':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r1,r2 = axiom
				inference_tmp = []
				for (h,t) in self.r_ht[r1]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 6])
						break
					if (t,h) not in self.r_ht[r2] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
						num += 1
						inference_tmp.append([h,r1,t, t,r2,h])
						#self.inverse2entailment[axiom_key].append([h,r1,t, t,r2,h])

				for entailment in inference_tmp:
					self.inverse2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'equivalent' or type =='subproperty':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r1,r2 = axiom
				inference_tmp = []
				for (h,t) in self.r_ht[r1]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 6])
						break
					if (h,t) not in self.r_ht[r2] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
						num += 1
						inference_tmp.append([h,r1,t, h,r2,t])

				for entailment in inference_tmp:
					self.equivalent2entailment[axiom_key].append(entailment)
					self.subproperty2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)


		if type == 'inferencechain1':
			self.inferencechain12entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (e, h) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.hr_t[(e, r3)]:
						if (h, t) not in self.r_ht[r1] and (
										self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
							num += 1
							inference_tmp.append([e, r2, h, e, r3, t, h, r1, t])
							#self.inferencechain12entailment[axiom_key].append([[e, r2, h, e, r3, t, h, r1, t]])


				for entailment in inference_tmp:
					self.inferencechain12entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain2':
			self.inferencechain22entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (e, h) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.tr_h[(e, r3)]:
						if (h, t) not in self.r_ht[r1] and (
										self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
							num += 1
							inference_tmp.append([e, r2, h, t, r3, e, h, r1, t])
							#self.inferencechain22entailment[axiom_key].append([[e, r2, h, t, r3, e, h, r1, t]])

				for entailment in inference_tmp:
					self.inferencechain22entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)


		if type == 'inferencechain3':
			self.inferencechain32entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (h, e) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.hr_t[(e, r3)]:
						if (h, t) not in self.r_ht[r1] and (
										self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
							num += 1
							inference_tmp.append([h, r2, e, e, r3, t, h, r1, t])


				for entailment in inference_tmp:
					self.inferencechain32entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain4':
			self.inferencechain42entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (h, e) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.tr_h[(e, r3)]:
						if (h, t) not in self.r_ht[r1] and (
										self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
							num += 1
							inference_tmp.append([h, r2, e, t, r3, e, h, r1, t])

				for entailment in inference_tmp:
					self.inferencechain42entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)
		return inference, num

	def _materialize(self, axioms, type=None, sparse=False):
		inference = []
		# axiom2entailment is a dict
		# with the all axioms in the axiom pool as keys
		# and all the entailments for each axiom as values
		axiom_list = axioms
		# print('axiom_list', axiom_list)
		length = len(axioms)
		max_entailments = 5000
		num = 0
		if length == 0:
			if type == 'reflexive':
				np.reshape(inference, [-1, 3])
			elif type == 'symmetric' or type == 'inverse' or type == 'equivalent' or type == 'subproperty':
				np.reshape(inference, [-1, 6])
			elif type == 'transitive' or type == 'inferencechain':
				np.reshape(inference, [-1, 9])
			else:
				raise NotImplementedError
			return inference, num

		if type == 'reflexive':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h, t) in self.r_ht[r]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 2])
						break
					if h != t: #and self.entity2sparsity[h] > self.sparsity:
						num += 1
						inference_tmp.append([h, r, h])
				for entailment in inference_tmp:
					self.reflexive2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'symmetric':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h, t) in self.r_ht[r]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 2])
						break
					if (t, h) not in self.r_ht[r]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
						num += 1
						inference_tmp.append([h, r, t, t, r, h])
				for entailment in inference_tmp:
					self.symmetric2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'transitive':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r = axiom[0]
				inference_tmp = []
				for (h, t) in self.r_ht[r]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					# (t,r,e) exist but (h,r,e) not exist and e!=h
					for e in self.hr_t[(t, r)] - self.hr_t[(h, r)]:
						if e != h: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
							num += 1
							inference_tmp.append([h, r, t, t, r, e, h, r, e])

				for entailment in inference_tmp:
					self.transitive2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inverse':
			# self.inverse2entailment = defaultdict(list)
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r1, r2 = axiom
				inference_tmp = []
				for (h, t) in self.r_ht[r1]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 6])
						break
					if (t, h) not in self.r_ht[r2]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
						num += 1
						inference_tmp.append([h, r1, t, t, r2, h])
				for entailment in inference_tmp:
					self.inverse2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'equivalent' or type == 'subproperty':
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				r1, r2 = axiom
				inference_tmp = []
				for (h, t) in self.r_ht[r1]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 6])
						break
					if (h, t) not in self.r_ht[r2]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
						num += 1
						inference_tmp.append([h, r1, t, h, r2, t])

				for entailment in inference_tmp:
					self.equivalent2entailment[axiom_key].append(entailment)
					self.subproperty2entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain1':
			self.inferencechain12entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (e, h) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.hr_t[(e, r3)]:
						if (h, t) not in self.r_ht[r1]:
							num += 1
							inference_tmp.append([e, r2, h, e, r3, t, h, r1, t])
				for entailment in inference_tmp:
					self.inferencechain12entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain2':
			self.inferencechain22entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (e, h) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.tr_h[(e, r3)]:
						if (h, t) not in self.r_ht[r1]:
							num += 1
							inference_tmp.append([e, r2, h, t, r3, e, h, r1, t])
				for entailment in inference_tmp:
					self.inferencechain22entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain3':
			self.inferencechain32entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				# print('%d/%d' % (i, length))
				r1, r2, r3 = axiom
				inference_tmp = []
				for (h, e) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.hr_t[(e, r3)]:
						if (h, t) not in self.r_ht[r1]:
							num += 1
							inference_tmp.append([h, r2, e, e, r3, t, h, r1, t])
				for entailment in inference_tmp:
					self.inferencechain32entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)

		if type == 'inferencechain4':
			self.inferencechain42entailment = defaultdict(list)
			i = 0
			for axiom in axiom_list:
				axiom_key = tuple(axiom)
				i += 1
				r1, r2, r3 = axiom
				inference_tmp = []
				for (h, e) in self.r_ht[r2]:
					if len(inference_tmp) > max_entailments:
						inference_tmp = np.reshape([], [-1, 9])
						break
					for t in self.tr_h[(e, r3)]:
						if (h, t) not in self.r_ht[r1]:
							num += 1
							inference_tmp.append([h, r2, e, t, r3, e, h, r1, t])
				for entailment in inference_tmp:
					self.inferencechain42entailment[axiom_key].append(entailment)
				inference.append(inference_tmp)
		return inference, num

	def _init_valid_axioms(self):
		# init valid axioms
		self.valid_reflexive, self.valid_symmetric, self.valid_transitive,\
		self.valid_inverse, self.valid_subproperty, self.valid_equivalent,\
		self.valid_inferencechain1, self.valid_inferencechain2, \
		self.valid_inferencechain3, self.valid_inferencechain4 = [[] for x in range(self.axiom_types)]

		# init valid axiom entailments
		self.valid_reflexive2entailment, self.valid_symmetric2entailment, self.valid_transitive2entailment, \
		self.valid_inverse2entailment, self.valid_subproperty2entailment, self.valid_equivalent2entailment, \
		self.valid_inferencechain12entailment, self.valid_inferencechain22entailment, \
		self.valid_inferencechain32entailment, self.valid_inferencechain42entailment = [[] for x in range(self.axiom_types)]

		# init valid axiom entailments probability
		self.valid_reflexive_p, self.valid_symmetric_p, self.valid_transitive_p, \
		self.valid_inverse_p, self.valid_subproperty_p, self.valid_equivalent_p, \
		self.valid_inferencechain1_p, self.valid_inferencechain2_p,\
		self.valid_inferencechain3_p, self.valid_inferencechain4_p= [[] for x in range(self.axiom_types)]

		# init valid axiom batchsize
		self.reflexive_batchsize = 1
		self.symmetric_batchsize = 1
		self.transitive_batchsize = 1
		self.inverse_batchsize = 1
		self.subproperty_batchsize = 1
		self.equivalent_batchsize = 1
		#self.inferencechain_batchsize = 1
		self.inferencechain1_batchsize = 1
		self.inferencechain2_batchsize = 1
		self.inferencechain3_batchsize = 1
		self.inferencechain4_batchsize = 1



	def update_valid_axioms(self, input):
		# this function is used to select high probability axioms as valid axioms
		# and record their scores
		valid_axioms = [self._select_high_probability(list(prob), axiom) for prob,axiom in zip(input, self.axiompool)]

		self.valid_reflexive, self.valid_symmetric, self.valid_transitive, \
		self.valid_inverse, self.valid_subproperty, self.valid_equivalent, \
		self.valid_inferencechain1, self.valid_inferencechain2, \
		self.valid_inferencechain3, self.valid_inferencechain4 = valid_axioms
		# update the batchsize of axioms and entailments
		self._reset_valid_axiom_entailment()

		'''
		logging.debug('the valid axioms after updated: %s'%(valid_axioms))
		logging.debug('the valid reflexive axioms: %s'%(self.valid_reflexive))
		logging.debug('the valid symmetric axioms: %s'%(self.valid_symmetric))
		logging.debug('the valid inverse axioms: %s'%(self.valid_inverse))
		logging.debug('the valid inferencevhain axioms: %s'%(self.valid_inferencechain))
		'''

	def _select_high_probability(self, prob, axiom):
		# select the high probability axioms and recore their probabilities
		valid_axiom = [[axiom[prob.index(p)],[p]] for p in prob if p>self.select_probability]
		return valid_axiom

	def reset(self, batch_size):
		self.batch_size = batch_size
		self.train_start = 0
		self.valid_start = 0
		self.test_start = 0
		self.num_train_triples = len(self.train_ids_labels)
		self.num_batch_train = round(self.num_train_triples/self.batch_size)


	def _reset_valid_axiom_entailment(self):

		self.infered_hr_t = defaultdict(set)
		self.infered_tr_h = defaultdict(set)

		self.valid_reflexive2entailment, self.valid_reflexive_p = \
			self._valid_axiom2entailment(self.valid_reflexive, self.reflexive2entailment)

		self.valid_symmetric2entailment, self.valid_symmetric_p = \
			self._valid_axiom2entailment(self.valid_symmetric, self.symmetric2entailment)

		self.valid_transitive2entailment, self.valid_transitive_p = \
			self._valid_axiom2entailment(self.valid_transitive, self.transitive2entailment)

		self.valid_inverse2entailment, self.valid_inverse_p = \
			self._valid_axiom2entailment(self.valid_inverse, self.inverse2entailment)

		self.valid_subproperty2entailment, self.valid_subproperty_p = \
			self._valid_axiom2entailment(self.valid_subproperty, self.subproperty2entailment)

		self.valid_equivalent2entailment, self.valid_equivalent_p = \
			self._valid_axiom2entailment(self.valid_equivalent, self.equivalent2entailment)


		self.valid_inferencechain12entailment, self.valid_inferencechain1_p = \
			self._valid_axiom2entailment(self.valid_inferencechain1, self.inferencechain12entailment)

		self.valid_inferencechain22entailment, self.valid_inferencechain2_p = \
			self._valid_axiom2entailment(self.valid_inferencechain2, self.inferencechain22entailment)

		self.valid_inferencechain32entailment, self.valid_inferencechain3_p = \
			self._valid_axiom2entailment(self.valid_inferencechain3, self.inferencechain32entailment)
		self.valid_inferencechain42entailment, self.valid_inferencechain4_p = \
			self._valid_axiom2entailment(self.valid_inferencechain4, self.inferencechain42entailment)


	def _valid_axiom2entailment(self, valid_axiom, axiom2entailment):
		valid_axiom2entailment = []
		valid_axiom_p = []
		for axiom_p in valid_axiom:
			axiom = tuple(axiom_p[0])
			p = axiom_p[1]
			for entailment in axiom2entailment[axiom]:
				valid_axiom2entailment.append(entailment)
				valid_axiom_p.append(p)
				h,r,t = entailment[-3:]
				self.infered_hr_t[(h,r)].add(t)
				self.infered_tr_h[(t,r)].add(h)
		return valid_axiom2entailment, valid_axiom_p

	def _generate_axiom_batchsize(self, valid_axiom2entailment):
		return round(len(valid_axiom2entailment)/self.num_batch_train)

	def generate_train_batch(self):
		origin_triples = self.train_ids_labels
		inject_triples = self.train_ids_labels_inject
		inject_num = self.inject_triple_percent*len(self.train_ids_labels)
		if len(inject_triples)> inject_num and inject_num >0:
			np.random.shuffle(inject_triples)
			inject_triples = inject_triples[:inject_num]
		train_triples = np.concatenate([origin_triples, inject_triples], axis=0)

		self.num_train_triples = len(train_triples)
		self.num_batch_train = round(self.num_train_triples/self.batch_size)
		print('self.num_batch_train', self.num_batch_train)
		np.random.shuffle(train_triples)
		for i in range(self.num_batch_train):
			t1 = time.time()
			start = i*self.batch_size
			end = min(start+self.batch_size, self.num_train_triples)
			positive = train_triples[start:end]
			t2 = time.time()
			yield positive

	def _generate_negative(self, positive):
		positive = np.asarray(positive)
		num = len(positive)
		replace_h, replace_t = [np.random.randint(self.num_entity, size=self.neg_samples*num) for i in range(2)]
		replace_r = np.random.randint(self.num_relation, size=self.neg_samples*num)

		neg_h, neg_r, neg_t = [np.copy(positive) for i in range(3)]
		neg_h[:, 0] = replace_h
		neg_r[:, 1] = replace_r
		neg_t[:, 2] = replace_t
		negative = np.concatenate((neg_h, neg_r, neg_t), axis=0)

		return negative

	def negative_triple_generator(self, input_postive_queue, output_queue):
		while True:
			dat = input_postive_queue.get()
			if dat == None:
				break
			positive = np.asarray(dat)
			replace_h, replace_t = [np.random.randint(self.num_entity, size=len(positive)*self.neg_samples) for i in range(2)]
			replace_r = np.random.randint(self.num_relation, size=len(positive)*self.neg_samples)
			neg_h, neg_r, neg_t = [np.copy(np.repeat(positive, self.neg_samples, axis=0)) for i in range(3)]
			neg_h[:, 0] = replace_h
			neg_r[:, 1] = replace_r
			neg_t[:, 2] = replace_t
			negative = np.concatenate((neg_h, neg_r, neg_t), axis=0)
			output_queue.put((positive, negative))


	def _specific_axiom_batch(self, iter,
							  valid_axiom2entailment,
							  valid_axiom_p, batchsize,
							  axiom2entailment):
		# this function is used to genetate a batch of entailments for specific axioms
		time1 = time.time()

		start = min(iter*batchsize, len(valid_axiom2entailment))
		end = min(start+batchsize, len(valid_axiom2entailment))
		if end>start:
			output_entailment = valid_axiom2entailment[start:end]
			output_p = valid_axiom_p[start:end]
		# this is used to avoid empty axiom batch
		else:
			output_entailment = []
			output_p = []

			entailments = list(axiom2entailment.values())

			output_entailment.append(random.choice(entailments[0]))
			output_p.append([0.0])
			logging.debug('output_axiom2entailment from artificial generation:%s, %s'%(str(output_entailment)[:200], output_p))



		if output_entailment == []:
			raise EOFError
		#print('time:', time1_1-time1, time1_2-time1_1, time1_3 - time1_2,  time2-time1_3, time3-time2)
		return output_entailment, output_p


	# add the new triples from axioms to training triple
	def update_train_triples(self, epoch=0, update_per = 10):
		reflexive_triples, symmetric_triples, transitive_triples, inverse_triples,\
			equivalent_triples, subproperty_triples, inferencechain1_triples, \
			inferencechain2_triples, inferencechain3_triples, inferencechain4_triples = [ np.reshape(np.asarray([]), [-1, 3]) for i in range(self.axiom_types)]
		reflexive_p, symmetric_p, transitive_p, inverse_p, \
			equivalent_p, subproperty_p, inferencechain1_p, \
			inferencechain2_p, inferencechain3_p, inferencechain4_p = [np.reshape(np.asarray([]), [-1, 1]) for i in
																		   range(self.axiom_types)]
		if epoch >= 20:
			print("len(self.valid_reflexive2entailment):", len(self.valid_reflexive2entailment))
			print("len(self.valid_symmetric2entailment):", len(self.valid_symmetric2entailment))
			print("len(self.valid_transitive2entailment)", len(self.valid_transitive2entailment))
			print("len(self.valid_inverse2entailment)", len(self.valid_inverse2entailment))
			print("len(self.valid_equivalent2entailment)", len(self.valid_equivalent2entailment))
			print("len(self.valid_subproperty2entailment)", len(self.valid_subproperty2entailment))

			valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment,\
			valid_inverse2entailment, valid_equivalent2entailment, valid_subproperty2entailment, \
			valid_inferencechain12entailment, valid_inferencechain22entailment,\
			valid_inferencechain32entailment, valid_inferencechain42entailment = [[] for i in range(10)]

			if len(self.valid_reflexive2entailment)>0:
				valid_reflexive2entailment = np.reshape(np.asarray(self.valid_reflexive2entailment), [-1, 3])
				reflexive_triples = np.asarray(valid_reflexive2entailment)[:, -3:]
				reflexive_p = np.reshape(np.asarray(self.valid_reflexive_p),[-1,1])

			if len(self.valid_symmetric2entailment) > 0:
				valid_symmetric2entailment = np.reshape(np.asarray(self.valid_symmetric2entailment), [-1, 6])
				symmetric_triples = np.asarray(valid_symmetric2entailment)[:, -3:]
				symmetric_p = np.reshape(np.asarray(self.valid_symmetric_p),[-1,1])

			if len(self.valid_transitive2entailment) > 0:
				valid_transitive2entailment = np.reshape(np.asarray(self.valid_transitive2entailment), [-1, 9])
				transitive_triples = np.asarray(valid_transitive2entailment)[:, -3:]
				transitive_p = np.reshape(np.asarray(self.valid_transitive_p), [-1, 1])

			if len(self.valid_inverse2entailment) > 0:
				valid_inverse2entailment = np.reshape(np.asarray(self.valid_inverse2entailment), [-1, 6])
				inverse_triples = np.asarray(valid_inverse2entailment)[:, -3:]
				inverse_p = np.reshape(np.asarray(self.valid_inverse_p), [-1, 1])

			if len(self.valid_equivalent2entailment) > 0:
				valid_equivalent2entailment = np.reshape(np.asarray(self.valid_equivalent2entailment), [-1, 6])
				equivalent_triples = np.asarray(valid_equivalent2entailment)[:, -3:]
				equivalent_p = np.reshape(np.asarray(self.valid_equivalent_p), [-1, 1])

			if len(self.valid_subproperty2entailment) > 0:
				valid_subproperty2entailment = np.reshape(np.asarray(self.valid_subproperty2entailment), [-1, 6])
				subproperty_triples = np.asarray(valid_subproperty2entailment)[:, -3:]
				subproperty_p = np.reshape(np.asarray(self.valid_subproperty_p),[-1,1])

			if len(self.valid_inferencechain12entailment) > 0:
				valid_inferencechain12entailment = np.reshape(np.asarray(self.valid_inferencechain12entailment), [-1, 9])
				inferencechain1_triples = np.asarray(valid_inferencechain12entailment)[:, -3:]
				inferencechain1_p = np.reshape(np.asarray(self.valid_inferencechain1_p), [-1, 1])

			if len(self.valid_inferencechain22entailment) > 0:
				valid_inferencechain22entailment = np.reshape(np.asarray(self.valid_inferencechain22entailment), [-1, 9])
				inferencechain2_triples = np.asarray(valid_inferencechain22entailment)[:, -3:]
				inferencechain2_p = np.reshape(np.asarray(self.valid_inferencechain2_p), [-1, 1])

			if len(self.valid_inferencechain32entailment) > 0:
				valid_inferencechain32entailment = np.reshape(np.asarray(self.valid_inferencechain32entailment), [-1, 9])
				inferencechain3_triples = np.asarray(valid_inferencechain32entailment)[:, -3:]
				inferencechain3_p = np.reshape(np.asarray(self.valid_inferencechain3_p), [-1, 1])

			if len(self.valid_inferencechain42entailment) > 0:
				valid_inferencechain42entailment = np.reshape(np.asarray(self.valid_inferencechain42entailment), [-1, 9])
				inferencechain4_triples = np.asarray(valid_inferencechain42entailment)[:, -3:]
				inferencechain4_p = np.reshape(np.asarray(self.valid_inferencechain4_p), [-1, 1])

			# pickle.dump(self.reflexive_entailments, open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'wb'))
			# store all the injected triples
			entailment_all = (valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment,
					 valid_inverse2entailment, valid_equivalent2entailment, valid_subproperty2entailment,
					 valid_inferencechain12entailment,valid_inferencechain22entailment,
							  valid_inferencechain32entailment,valid_inferencechain42entailment)
			pickle.dump(entailment_all, open(os.path.join(self.axiom_dir, 'valid_entailments.pickle'), 'wb'))



		train_inject_triples = np.concatenate([reflexive_triples, symmetric_triples, transitive_triples, inverse_triples,
												equivalent_triples, subproperty_triples, inferencechain1_triples,
											   inferencechain2_triples,inferencechain3_triples,inferencechain4_triples],
												axis=0)

		train_inject_triples_p = np.concatenate([reflexive_p,symmetric_p, transitive_p, inverse_p,
											   equivalent_p, subproperty_p, inferencechain1_p,
												 inferencechain2_p,inferencechain3_p,inferencechain4_p],
											  axis=0)

		self.train_inject_triples = train_inject_triples
		inject_labels = np.reshape(np.ones(len(train_inject_triples)), [-1, 1]) * self.axiom_weight * train_inject_triples_p
		train_inject_ids_labels = np.concatenate([train_inject_triples, inject_labels],
												axis=1)


		self.train_ids_labels_inject = train_inject_ids_labels


		print('num reflexive triples', len(reflexive_triples))
		print('num symmetric triples', len(symmetric_triples))
		print('num transitive triples', len(transitive_triples))
		print('num inverse triples', len(inverse_triples))
		print('num equivalent triples', len(equivalent_triples))
		print('num subproperty triples', len(subproperty_triples))
		print('num inferencechain1 triples', len(inferencechain1_triples))
		print('num inferencechain2 triples', len(inferencechain2_triples))
		print('num inferencechain3 triples', len(inferencechain3_triples))
		print('num inferencechain4 triples', len(inferencechain4_triples))





















