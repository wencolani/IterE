import numpy as np
from collections import defaultdict
import pickle
import os
import timeit
import argparse

# =========== NOTES ======================
# This riginal version should work if you 
# put it inside the dataset dictionary
# ========================================

# input: triple file
# output: triples [(h,r,t),...]
def readtrainfile(file):
	triples = []
	hr_t = defaultdict(set)
	ht_r = defaultdict(set)
	h_t = defaultdict(set)
	t_h = defaultdict(set)
	r_hrt = defaultdict(list)
	f = open(file, 'r')
	for line in f.readlines():
		h, t, r = line.strip().split('\t')
		#assert len(line_list) == 3
		triples.append((h,r,t))
		hr_t[(h,r)].add(t)
		ht_r[(h,t)].add(r)
		h_t[h].add(t)
		t_h[t].add(h)
		r_hrt[r].append((h,r,t))
	return triples, hr_t, ht_r, h_t, t_h, r_hrt

def pickledump(*inputs):
	num,  reflexive, symmetric, transitive, inverse, equivalent, subProperty, inferenceChain = inputs
	with open('./output/reflexive_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(reflexive, f)
	with open('./output/symmetric_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(symmetric, f)
	with open('./output/transitive_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(transitive, f)
	with open('./output/inverse_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(inverse, f)
	with open('./output/equivalent_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(equivalent, f)
	with open('./output/subProperty_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(subProperty, f)
	with open('./output/inferenceChain_'+str(num)+'.pickle', 'wb') as f:
		pickle.dump(inferenceChain, f)


# input: triples
# output: possible axioms for each type
# Axioms(dict): {inverse:{...},symmetric: {...}, ... }
def generateAxioms(triple_data, p, g, dataset_dir):
	num_samples = 100
	triples, hr_t, ht_r, h_t, t_h, r_hrt = triple_data
	num_axiom_types = 7
	reflexive, symmetric, transitive, equivalent, inverse, \
	subProperty, inferenceChain = [ set() for i in range(num_axiom_types)]
	count = 0

	for rel in r_hrt.keys():
		# the number of triples about r to generate axioms
		N = len(r_hrt[rel])
		pN = p*N
		num_samples = round(N - N*pow(1-g,1/pN))
		np.random.shuffle(r_hrt[rel])
		num_triples = min(num_samples, len(r_hrt[rel]))
		print("num_triples", num_triples)
		hrts = r_hrt[rel][:num_triples]

		if count % 1 == 0:
			print(
				'num:%d / reflexive:%d / symmetric:%d / transitive:%d / inverse:%d / equivalent: %d / subProperty: %d / inferenceChain: %d'
				% (
				count, len(reflexive), len(symmetric), len(transitive), len(inverse), len(equivalent), len(subProperty),
				len(inferenceChain)))
		'''
		if count % 100 == 0:
			pickledump(count, reflexive, symmetric, transitive, inverse, equivalent, subProperty, inferenceChain)
		'''
		count_triples = 0
		for h,r,t in hrts:
			print(count_triples, end='\r')
			count_triples += 1
			# 1 relexive
			if h == t:
				reflexive.add((r,))

			# 2 symmetric
			if (t,r,h) in r_hrt[r]:
				symmetric.add((r,))

			# 3 transitive
			for t_tmp in hr_t[(h, r)]:
				if t_tmp != t and (t_tmp, r, t) in r_hrt[r]:
					transitive.add((r,))

			# 4 equivalent and 6 subProperty
			for r_tmp in ht_r[(h, t)]:
				if r_tmp != r:
					equivalent.add((r, r_tmp))
					subProperty.add((r, r_tmp))

			# 5 inverse
			if (t, h) in ht_r.keys():
				for r_tmp in ht_r[(t, h)]:
					inverse.add((r, r_tmp))

			# 7 inferenceChain
			# h --> e --> t
			h_e = h_t[h]
			t_e = t_h[t]
			e_common = h_e.intersection(t_e)
			for e in e_common:
				for r1 in ht_r[(h, e)]:
					for r2 in ht_r[(e, t)]:
						inferenceChain.add((r, r1, r2))
		count += 1
		'''
		if count > 3:
			break
		'''

	print('finish processing')
	
	print('write reflexive file')
	writefile(reflexive, os.path.join(dataset_dir, 'axiom_pool/axiom_reflexive.txt'), 1)
	print('write symmetric file')
	writefile(symmetric, os.path.join(dataset_dir, 'axiom_pool/axiom_symmetric.txt'), 1)
	print('write transitive file')
	writefile(transitive, os.path.join(dataset_dir, 'axiom_pool/axiom_transitive.txt'), 1)
	print('write inverse file')
	writefile(inverse, os.path.join(dataset_dir, 'axiom_pool/axiom_inverse.txt'), 2)
	print('write equivalent file')
	writefile(equivalent, os.path.join(dataset_dir, 'axiom_pool/axiom_equivalent.txt'), 2)
	print('write subProperty file')
	writefile(subProperty, os.path.join(dataset_dir, 'axiom_pool/axiom_subProperty.txt'), 2)
	print('write inferenceChain file')
	writefile(inferenceChain, os.path.join(dataset_dir, 'axiom_pool/axiom_inferenceChain.txt'), 3)


def writefile(axioms, file, num_element):
	with open(file, 'w') as f:
		for obj in axioms:
			for i in range(num_element):
				f.write(obj[i])
				if i == num_element-1:
					f.write('\n')
				else:
					f.write('\t')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Experiment setup')
	# misc(short for miscellaneous)
	parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default='./dataset/FB15k')
	parser.add_argument('--train_file', dest='train_file', type=str, default='train.txt')
	parser.add_argument('--axiom_probability', dest='axiom_probability', type=float, default=0.5)
	parser.add_argument('--axiom_proportion', dest='axiom_proportion', type=float, default=0.95)
	# dest for parser
	option = parser.parse_args()

	file_train = os.path.join(option.dataset_dir,option.train_file)
	start = timeit.default_timer()
	# keep the axioms with probability larger than p
	p = option.axiom_probability
	# the probability of keep axioms when generating
	g = option.axiom_proportion
	# triples: [(head, rel, tail), ...]
	triple_data = readtrainfile(file_train)
	generateAxioms(triple_data, p, g, option.dataset_dir)
	end = timeit.default_timer()
	print('cost time:', end-start)