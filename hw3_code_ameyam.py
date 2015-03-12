# Computational linguistics HW 3
# Author Ameya More
# Pennkey: ameyam


from nltk import sent_tokenize, word_tokenize
from liblinearutil import *
import os
from collections import Counter
from BeautifulSoup import BeautifulSoup
import xml.etree.ElementTree as ET
import operator
import math
import pickle

def ameyam_get_all_files(directory):
        directory = directory + '/'
        file_list = []
        for root, dirs, files in os.walk(directory, topdown=True):
                for name in files:
                        absolute_path = (os.path.join(root, name))
                        file_list.append(absolute_path)
        return file_list


def ameyam_preprocess(directory, corenlp_output):

	file_list = ameyam_get_all_files(directory)
	file_list_file = open('file_list_preprocess.txt', 'w')
	for file_name in file_list:
		file_list_file.write(file_name + '\n')
	file_list_file.close()
	os.system('java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar \
-Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse \
-filelist file_list_preprocess.txt ' + \
'-outputDirectory ' + corenlp_output)

def extract_top_words(xml_directory):
	file_list = ameyam_get_all_files(xml_directory)
	top_2000 = []
	words = []
	for file_name in file_list:
		words = words + ameyam_get_words_from_xml(file_name)
	word_freq = Counter(words)
#	print word_freq
	
	sorted_x = sorted(word_freq.items(), key=operator.itemgetter(1))
	j = len(sorted_x) - 1
	for i in range(0, 2000):
		if(j >= 0):
			top_2000.append(sorted_x[j][0])
			j = j - 1
		else:
			break
	return top_2000

def ameyam_get_words_from_xml(input_xml):
        tree = ET.parse(input_xml)
        list_all_words = []
        root = tree.getroot()
        for neighbor in root.iter('token'):
                word = neighbor[0].text
                list_all_words.append(word.lower())
        return list_all_words


def ameyam_get_file_dependencies(input_xml):
        tree = ET.parse(input_xml)
        list_all_words = []
        root = tree.getroot()
        for neighbor in root.iter('basic-dependencies'):
                for dependency in neighbor.iter('dep'):
                        relation = dependency.attrib['type']
                        governer = dependency[0].text.lower()
                        dependent = dependency[1].text.lower()
                        dep_tuple = (relation,governer,dependent)
                        list_all_words.append(dep_tuple)
        return list_all_words



def map_unigrams(xml_filename, top_words):
	words_xml = ameyam_get_words_from_xml(xml_filename)
	feature_vector = []
	for word in top_words:
		if word in words_xml:
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	return feature_vector

def ameyam_extra_credit_load_feature_vectors():
	vector_file = open('extra_credit_word2vec.txt', 'r')
	vectors = vector_file.readlines()
	vector_dict = {}
	for vector in vectors:
		vector = vector.split()
		temp = vector[1:]
		temp = [ float(x) for x in temp ]
		vector_dict[vector[0]] = temp
	return vector_dict




def ameyam_load_feature_vectors():
	vector_file = open('/project/cis/nlp/tools/word2vec/vectors.txt', 'r')
	vectors = vector_file.readlines()
	vector_dict = {}
	for vector in vectors:
		vector = vector.split()
		temp = vector[1:]
		temp = [ float(x) for x in temp ]
		vector_dict[vector[0]] = temp
	return vector_dict

def cosine_similarity(x, y):
	dot_product = 0
	for i in range(0, len(x)):
		dot_product = dot_product + (x[i] * y[i])

	mag_x = 0
	mag_y = 0

	for i in range(0, len(x)):
		mag_x = mag_x + (x[i] * x[i])
		mag_y = mag_y + (y[i] * y[i])

	mag_x = math.sqrt(mag_x)
	mag_y = math.sqrt(mag_y)

#TODO:Determine if divide by zero check is required. Fucked up here last time.
	cosine = float(dot_product)/(float(mag_x)*float(mag_y))

	return cosine

def extra_credit_extract_similarity(top_words):
	vector_dict = ameyam_extra_credit_load_feature_vectors()
	similarity_dict = {}
	for word1 in top_words:
		similarity_dict[word1] = {}
		for word2 in top_words:
			if(word1 == word2):
				similarity_dict[word1][word2] = 1.0
				continue
			if((word1 not in vector_dict) or (word2 not in vector_dict)):
				continue
			if(word2 in similarity_dict):
				if(word1 in similarity_dict[word2]):
					similarity = similarity_dict[word2][word1]
				else:
					similarity = 0
			else:	
				similarity = cosine_similarity(vector_dict[word1], vector_dict[word2])
			if(similarity != 0):
				similarity_dict[word1][word2] = similarity

	return similarity_dict

def extract_similarity(top_words):
	vector_dict = ameyam_load_feature_vectors()
	similarity_dict = {}
	for word1 in top_words:
		similarity_dict[word1] = {}
		for word2 in top_words:
			if(word1 == word2):
				similarity_dict[word1][word2] = 1.0
				continue
			if((word1 not in vector_dict) or (word2 not in vector_dict)):
				continue
			if(word2 in similarity_dict):
				if(word1 in similarity_dict[word2]):
					similarity = similarity_dict[word2][word1]
				else:
					similarity = 0
			else:	
				similarity = cosine_similarity(vector_dict[word1], vector_dict[word2])
			if(similarity != 0):
				similarity_dict[word1][word2] = similarity

	return similarity_dict

"""
def extract_similarity(top_words):
	print 'started loading similarity vectors'
	vector_dict = ameyam_load_feature_vectors()
	print 'finished loading similarity vectors'
	similarity_dict = {}
	for word1 in top_words:
		similarity_dict[word1] = {}
		for word2 in top_words:
			if(word1 == word2):
				similarity_dict[word1][word2] = 1.0
				continue
			if((word1 not in vector_dict) or (word2 not in vector_dict)):
				continue
			similarity = cosine_similarity(vector_dict[word1], vector_dict[word2])
			if(similarity != 0):
				similarity_dict[word1][word2] = similarity

	return similarity_dict
"""
def map_expanded_unigrams(xml_file, top_words, similarity_matrix):
	feature_vector = map_unigrams(xml_file, top_words)
	enriched_vector = []
	outer_index = 0
	for element in feature_vector:
		if(element == 1):
			enriched_vector.append(1.0)
		else:
			outer_word = top_words[outer_index]
			inner_index = 0
			max_similarity = -100
			for element2 in feature_vector:
				if(element2 == 1):
					inner_word = top_words[inner_index]
					if outer_word in similarity_matrix.keys():
						temp = similarity_matrix[outer_word]
						if inner_word in temp:
							similarity = similarity_matrix[outer_word][inner_word]
						else:
							similarity = 0
					if(similarity > max_similarity):
						max_similarity = similarity
				inner_index = inner_index + 1
			enriched_vector.append(max_similarity)
		outer_index = outer_index + 1
	return enriched_vector


def extract_top_dependencies(xml_directory):
	file_list = ameyam_get_all_files(xml_directory)
	dependencies = []
	for file_name in file_list:
		dependencies = dependencies + ameyam_get_file_dependencies(file_name)
#		print dependencies
	dependencies_freq = Counter(dependencies)
	sorted_x = sorted(dependencies_freq.items(), key=operator.itemgetter(1))
	j = len(sorted_x) - 1
	top_2000 = []
	for i in range(0, 2000):
		if(j >= 0):
			top_2000.append(sorted_x[j][0])
			j = j - 1
		else:
			break
	return top_2000


def map_dependencies(xml_filename, dependency_list):
	dependencies_in_xml = ameyam_get_file_dependencies(xml_filename)
	dependency_vector = []
	for dependency in dependency_list:
		if dependency in dependencies_in_xml:
			dependency_vector.append(1)
		else:
			dependency_vector.append(0)
	return dependency_vector

def ameyam_get_prod_rules_for_file(file_name):
        tree = ET.parse(file_name)
        list_all_words = []
        root = tree.getroot()
	stack = []
	temp_dict = {}
	prod_rules = []
	for parse_tree in root.iter('parse'):
#		print 'parse tree:' + parse_tree.text + '\n'
		parse_tree = parse_tree.text
		parse_tree = parse_tree.split()
#		print parse_tree
		for token in parse_tree:
#			print stack
#			print temp_dict
#			print prod_rules
#			print '\n'
			if token[0] == '(':
#				print 'Pushing ' + token[1:]
#				print stack
				if token[1:] not in stack:
					stack.append(token[1:])
					temp_dict[token[1:]] = []
				else:
					i = 1
					while(True):
						temp_token = token[1:] + '_' + str(i) + '_FLAGGED'
						if temp_token not in stack:
							break
						i = i + 1
					stack.append(temp_token)
					temp_dict[temp_token] = []
			elif token[len(token) - 1] == ')':
#				print 'Popping ' + token
				for i in range(0, len(token)):
					if(token[i] == ')'):
						break
				length = len(token)
				for j in range(i, length):
#					print 'before popping:'
#					print stack
#					print '\n'
					top_token = stack.pop()
#					print 'after popping'
#					print stack
#					print '\n'
					if(len(stack) != 0 ):
						parent = stack[len(stack) - 1]
						temp_dict[parent].append(top_token)
					this_rule_list = temp_dict[top_token]
					temp_dict.pop(top_token, None)
					if (len(this_rule_list) != 0):
						temp = top_token.split('_')
						top_token = temp[0]
						this_rule = top_token
						
						for rhs in this_rule_list:
							temp = rhs.split('_')
							rhs = temp[0]
							this_rule = this_rule + '_' + rhs
						prod_rules.append(this_rule)
	return prod_rules


def extract_prod_rules(xml_directory):
	prod_rules = []
	file_list = ameyam_get_all_files(xml_directory)
	for file_name in file_list:
		prod_rules = prod_rules + ameyam_get_prod_rules_for_file(file_name)

	prod_freq = Counter(prod_rules)
	sorted_x = sorted(prod_freq.items(), key=operator.itemgetter(1))
	j = len(sorted_x) - 1
	top_2000 = []
	for i in range(0, 2000):
		if(j >= 0):
			top_2000.append(sorted_x[j][0])
			j = j - 1
		else:
			break
	return top_2000


def map_prod_rules(xml_filename, rules_list):
	prod_rules_in_file = ameyam_get_prod_rules_for_file(xml_filename)
	vector = []
	for rule in rules_list:
		if rule in prod_rules_in_file:
			vector.append(1)
		else:
			vector.append(0)
	return vector


#TODO: IMplement the last part ie liblinear_5.txt Also this function is taking way too long.
#No comments. It was tough to code. It should be tough to read.
def process_corpus(xml_dir, top_words, similarity_matrix, top_dependencies, prod_rules):
	if(xml_dir.find('test') != -1):
		binary_lexical = open('test_1.txt', 'w')
		lexical_expansion = open('test_2.txt', 'w')
		binary_dependency = open('test_3.txt', 'w')
		binary_prod_rules = open('test_4.txt', 'w')
		all_except_expanded = open('test_5.txt', 'w')
	else:
		binary_lexical = open('train_1.txt', 'w')
		lexical_expansion = open('train_2.txt', 'w')
		binary_dependency = open('train_3.txt', 'w')
		binary_prod_rules = open('train_4.txt', 'w')
		all_except_expanded = open('train_5.txt', 'w')

	file_list = ameyam_get_all_files(xml_dir)
	for filename in file_list:
		unigram_vector = map_unigrams(filename, top_words)
		expanded_unigram_vector = map_expanded_unigrams(filename, top_words, similarity_matrix)
		dependency_vector = map_dependencies(filename, top_dependencies)
		prod_rules_vector = map_prod_rules(filename, prod_rules)
		all_except_expanded_index = 1

#		print 'map_prod_rules done'
#		print unigram_vector
#		print '\n'
#		print expanded_unigram_vector
#		print '\n'
#		print dependency_vector
#		print '\n'
#		print prod_rules_vector

		all_except_expanded.write(os.path.basename(filename))
		binary_lexical.write(os.path.basename(filename))
		index = 1
		for element in unigram_vector:
			if(element > 0):
				binary_lexical.write(' ' + str(index) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_index) + ':' + str(element))
			all_except_expanded_index = all_except_expanded_index + 1
			index = index + 1
		binary_lexical.write('\n')

		lexical_expansion.write(os.path.basename(filename))
		index = 1
		for element in expanded_unigram_vector:
			if(element > 0):
				lexical_expansion.write(' ' + str(index) + ':' + str(element))
			index = index + 1
		lexical_expansion.write('\n')

		binary_dependency.write(os.path.basename(filename))
		index = 1
		for element in dependency_vector:
			if(element > 0):
				binary_dependency.write(' ' + str(index) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_index) + ':' + str(element))
			all_except_expanded_index = all_except_expanded_index + 1
			index = index + 1
		binary_dependency.write('\n')

		binary_prod_rules.write(os.path.basename(filename))
		index = 1
		for element in prod_rules_vector:
			if(element > 0):
				binary_prod_rules.write(' ' + str(index) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_index) + ':' + str(element))
			all_except_expanded_index = all_except_expanded_index + 1
			index = index + 1
		binary_prod_rules.write('\n')
		all_except_expanded.write('\n')


	lexical_expansion.close()
	binary_dependency.close()
	binary_prod_rules.close()
	all_except_expanded.close()


def ameyam_create_feature_genere_file(feature_set, genere, train_test):
	filename = open(genere + '_' + train_test + '_' + feature_set + '.txt', 'w')
	input_file = open(train_test + '_' + feature_set + '.txt', 'r')

	input_file = input_file.readlines()

	for file_vector in input_file:
		vector = file_vector.split()
		file_name = vector[0]
		if(file_name.startswith(genere)):
			filename.write('1')
			for i in range(1, len(vector)):
				filename.write( ' ' + vector[i])
			filename.write('\n')
		else:
			filename.write('-1')
			for i in range(1, len(vector)):
				filename.write( ' ' + vector[i])
			filename.write('\n')
	filename.close()

def ameyam_generate_40_files():
	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Computers', 'train')
	
	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Finance', 'train')

	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Health', 'train')

	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Research', 'train')

	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Computers', 'test')
	
	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Finance', 'test')

	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Health', 'test')

	for feature in range(1, 6):
		ameyam_create_feature_genere_file(str(feature), 'Research', 'test')

def ameyam_get_precision(predicted_labels, actual_labels, label):
	numerator = 0
	denominator = 0
	index = 0
	for predicted_label in predicted_labels:
		if(predicted_label == label):
			denominator = denominator + 1
			if(predicted_label == actual_labels[index]):
				numerator = numerator + 1
		index = index + 1
	precision = float(numerator)/float(denominator)
	return precision

def ameyam_get_recall(predicted_labels, actual_labels, label):
	numerator = 0
	denominator = 0
	index = 0
	for actual_label in actual_labels:
		if(actual_label == label):
			denominator = denominator + 1
			if(predicted_labels[index] == actual_label):
				numerator = numerator + 1
		index = index + 1
	recall = float(numerator)/float(denominator)
	return recall

def ameyam_get_fmeasure(precision, recall):
	return 2*precision*recall/(precision+recall)

def ameyam_get_weight(f,key):
	label = 0
	total = 0
	lines = open(f).readlines()
	for line in lines:
		if line.split()[0]==key:
			label+=1
		total+=1
	return float(label)/float(total)


def run_classifier(train_file, test_file):
	output_tuple = ()
	y,x = svm_read_problem(train_file)
	#no of lines starting from 1 
	weight2 = ameyam_get_weight(train_file,"1")
	weight1 = 1.0 - weight2

	model = train(y,x,'-s 0 -w1 '+ str(weight1) +' -w-1 '+ str(weight2))
    	y, x = svm_read_problem(test_file)
    	p_labs, p_acc, p_vals = predict(y,x, model,'-b 1')
    	pos_precision = ameyam_get_precision(p_labs,y,1)
    	pos_recall = ameyam_get_recall(p_labs,y,1)
#	pos_precision = 1
#	pos_recall = 1
    	pos_fmeasure = ameyam_get_fmeasure(pos_precision,pos_recall)
    	neg_precision = ameyam_get_precision(p_labs,y,-1)
   	neg_recall = ameyam_get_recall(p_labs,y,-1)
#	neg_precision = 1
#	neg_recall = 1
    	neg_fmeasure = ameyam_get_fmeasure(neg_precision,neg_recall)
    	measures = (pos_precision, pos_recall, pos_fmeasure, neg_precision, neg_fmeasure, neg_recall, p_acc[0])
    	prob_list = []
    	#TO DO check if you have to look at model.label
    	for sublist in p_vals:
		prob_list.append(sublist[0])
    	output_tuple = (p_labs, measures,prob_list)
    	return output_tuple

#Assumption here is that the 40 train and test files are already present
def ameyam_generate_results():
	results = open('results.txt', 'w')
	generes = ['Computers', 'Finance', 'Health', 'Research']
	for genere in generes:
		for i in range(1,6):
			output_tuple = run_classifier(genere + '_train_' + str(i) + '.txt', genere + '_test_' + str(i) + '.txt')
			measures = output_tuple[1]
			for measure in measures:
				results.write(str(measure) + ' ')
			results.write(genere + ':' + str(i) + '\n')
	results.close()

def save_object(obj, filename):
     filehandler = open(filename, 'wb')
     pickle.dump(obj, filehandler, pickle.HIGHEST_PROTOCOL)

#function to deserialize objects
def load_object(filename):
	file_handler = open(filename,'rb')
	object_file = pickle.load(file_handler)
	file_handler.close()
	return object_file

def ameyam_extra_credit_generate_classify_documents_result():
	generes = ['Health', 'Computers', 'Research', 'Finance']
	genere_probabilites = {}
	for genere in generes:
		tup = run_classifier(genere + '_train_6.txt', genere + '_test_6.txt')
		genere_probabilites[genere] = tup[2]

	p_labels = classify_documents(genere_probabilites['Health'], genere_probabilites['Computers'], genere_probabilites['Research'], genere_probabilites['Finance'])
	cnt = 0
	actual_labels = ameyam_get_all_files("xml_test_data")
	total = len(actual_labels)
	for i in range(0,total):
		if p_labels[i] in actual_labels[i]:
			cnt+=1
	acc = (float(cnt)/float(total))*(100.0)
	open('extra_credit_results.txt','a').write(str(acc))
	return


def ameyam_generate_classify_documents_result():
	results = open('results.txt', 'r')
	results = results.readlines()
	max_accuracies = {'Computers':[-1,-1], 'Finance':[-1,-1], 'Research':[-1,-1], 'Health':[-1,-1]}
	index = 0
	for genere in range(0, 4):
		this_result = results[index].split()
		this_genere = this_result[7]
		this_genere = this_genere.split(':')
		this_genere = this_genere[0]
		for feature in range(0,5):
			this_result = results[index].split()
			accuracy = this_result[6]
			if(max_accuracies[this_genere][0] < accuracy):
				max_accuracies[this_genere][0] = accuracy
				max_accuracies[this_genere][1] = feature
			index = index + 1
	features_giving_max_accuracy = []
	for genere in max_accuracies:
		features_giving_max_accuracy.append(max_accuracies[genere][1])
	max_accuracy_features_freq = Counter(features_giving_max_accuracy)
	#(feature, frequency)
	max_accuracy_feature = ['1', -1]
	for feature in max_accuracy_features_freq:
		if(max_accuracy_features_freq[feature] > max_accuracy_feature[1]):
			max_accuracy_feature[0] = feature
			max_accuracy_feature[1] = max_accuracy_features_freq[feature]
	generes = ['Health', 'Computers', 'Research', 'Finance'] 
	genere_probabilites = {}
	for genere in generes:
		tup = run_classifier(genere + '_train_' + str(max_accuracy_feature[0]) + '.txt', genere + '_test_' + str(max_accuracy_feature[0]) + '.txt')
		genere_probabilites[genere] = tup[2]
	p_labels = classify_documents(genere_probabilites['Health'], genere_probabilites['Computers'], genere_probabilites['Research'], genere_probabilites['Finance'])
	cnt = 0
	actual_labels = ameyam_get_all_files("xml_test_data")
	total = len(actual_labels)
	for i in range(0,total):
		if p_labels[i] in actual_labels[i]:
			cnt+=1
	acc = (float(cnt)/float(total))*(100.0)
	open("results.txt","a").write(str(acc))


def classify_documents(health_prob, computers_prob, research_prob, finance_prob):
   	p_labels = [];
    	for index in range(0,len(health_prob)):
		if ((health_prob[index] >= computers_prob[index]) and (health_prob[index] >= research_prob[index]) and (health_prob[index] >= finance_prob[index])):
			p_labels.append("Health")
		elif ((computers_prob[index] >= health_prob[index]) and (computers_prob[index] >= research_prob[index]) and (computers_prob[index] >= finance_prob[index])):
			p_labels.append("Computers")
		elif ((research_prob[index] >= health_prob[index]) and (research_prob[index] >= computers_prob[index]) and (research_prob[index] >= finance_prob[index])):
			p_labels.append("Research")
		else:
			p_labels.append("Finance")
	return p_labels


def ameyam_extra_credit_train_word2vec():
        fd = open('extra_credit_words.txt','w')
        train_files = ameyam_get_all_files('/home1/c/cis530/hw3/data')
        total_words = []
        for f in train_files:
                train_file=open(f,'r')
                for line in train_file:
                        w = [word for sent in sent_tokenize(line) for word in word_tokenize(sent)]
                        tokenized_list = ' '.join(w).strip('[]')
                        fd.write(tokenized_list)
                        fd.write(" ")
                train_file.close()
        fd.close()
    #remove punctuations
#        os.system("cat extra_credit_words.txt | tr -d '[:punct:]' > word_file1.txt")
        cmd = "/project/cis/nlp/tools/word2vec/word2vec -train extra_credit_words.txt -output extra_credit_word2vec.txt -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12"
        os.system(cmd)


def ameyam_extra_credit_genereate_8_files():
	generes = ['Computers', 'Research', 'Health', 'Finance']
	for genere in generes:
		ameyam_create_feature_genere_file('6', genere, 'test')
		ameyam_create_feature_genere_file('6', genere, 'train')

def ameyam_extra_credit_process_corpus(xml_dir, similarity_matrix):
	if(xml_dir.find('test') != -1):
		extra_credit_similarity = open('test_6.txt', 'w')
	else:
		extra_credit_similarity = open('train_6.txt', 'w')

	file_list = ameyam_get_all_files(xml_dir)
	for filename in file_list:
		expanded_unigram_vector = map_expanded_unigrams(filename, top_words, similarity_matrix)

		extra_credit_similarity.write(os.path.basename(filename))
		index = 1
		for element in expanded_unigram_vector:
			if(element > 0):
				extra_credit_similarity.write(' ' + str(index) + ':' + str(element))
			index = index + 1
		extra_credit_similarity.write('\n')

	extra_credit_similarity.close()

def ameyam_extra_credit_generate_results():
	results = open('extra_credit_results.txt', 'w')
	generes = ['Computers', 'Finance', 'Health', 'Research']
	for genere in generes:
		output_tuple = run_classifier(genere + '_train_6.txt', genere + '_test_6.txt')
		measures = output_tuple[1]
		for measure in measures:
			results.write(str(measure) + ' ')
		results.write(genere + ':6' + '\n')
	results.close()





#print ameyam_get_file_dependencies('xml_dir/Research_2005_01_02_1638819.txt.xml')

#os.system('java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar \
#-Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse \
#-filelist file_list.txt ' + \
#'-outputDirectory ./xml_dir')
#top_words = extract_top_words('/home1/c/cis530/hw3/xml_data')

"""
if os.path.isfile('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_words.txt'):
	print 'Loading training top words from file'
	top_words = load_object('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_words.txt')
else:
	top_words = extract_top_words('/home1/c/cis530/hw3/xml_data')
	save_object(top_words,'/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_words.txt')
print '##### TOP WORDS #####'
"""
#print top_words


#print map_unigrams('xml_dir/Research_2005_01_02_1638819.txt.xml', top_words)
"""
if os.path.isfile('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/similarity_matrix.txt'):
	similarity_matrix = load_object('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/similarity_matrix.txt')
else:
	print 'Loading similarity matrix from file'
	similarity_matrix = extract_similarity(top_words)
	save_object(similarity_matrix, '/home1/a/ameyam/computational_linguistics/hw3/saved_objects/similarity_matrix.txt')
print '###### SIMILARITY MATRIX ######'
#print similarity_matrix
"""
"""
if os.path.isfile('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/extra_credit_similarity_matrix.txt'):
	similarity_matrix = load_object('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/extra_credit_similarity_matrix.txt')
else:
	print 'Loading similarity matrix from file'
	similarity_matrix = extra_credit_extract_similarity(top_words)
	save_object(similarity_matrix, '/home1/a/ameyam/computational_linguistics/hw3/saved_objects/extra_credit_similarity_matrix.txt')
print '###### SIMILARITY MATRIX ######'
"""

#print map_expanded_unigrams('xml_test/Research_2005_01_02_1638819.txt.xml', top_words, similarity_dict)

#top_dependencies =  extract_top_dependencies('/home1/c/cis530/hw3/xml_data')

"""
if os.path.isfile('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_dependencies.txt'):
	print 'Loading top dependencies from file'
	top_dependencies = load_object('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_dependencies.txt')
else:
	top_dependencies =  extract_top_dependencies('/home1/c/cis530/hw3/xml_data')
	save_object(top_dependencies, '/home1/a/ameyam/computational_linguistics/hw3/saved_objects/top_dependencies.txt')
print '##### TOP DEPENDENCIES ######'
#print top_dependencies
"""

#prod_rules = extract_prod_rules('/home1/c/cis530/hw3/xml_data')
"""
if os.path.isfile('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/prod_rules.txt'):
	print 'Loading production rules from file'
	prod_rules = load_object('/home1/a/ameyam/computational_linguistics/hw3/saved_objects/prod_rules.txt')
else:
	prod_rules = extract_prod_rules('/home1/c/cis530/hw3/xml_data')
	save_object(prod_rules, '/home1/a/ameyam/computational_linguistics/hw3/saved_objects/prod_rules.txt')
print '##### PROD RULES #####'
"""
#print prod_rules
#print map_prod_rules('xml_dir/Research_2005_01_02_1638819.txt.xml', prodrules)
#process_corpus('/home1/c/cis530/hw3/xml_data', top_words, similarity_matrix, top_dependencies, prod_rules)
#process_corpus('/home1/a/ameyam/computational_linguistics/hw3/xml_test_data', top_words, similarity_matrix, top_dependencies, prod_rules)
#ameyam_generate_40_files()
#print run_classifier('Computers_and_the_Internet_test_1.txt', 'Computers_and_the_Internet_train_1.txt')
"""
ameyam_extra_credit_process_corpus('xml_test_data', similarity_matrix)
ameyam_extra_credit_process_corpus('/home1/c/cis530/hw3/xml_data', similarity_matrix)
ameyam_extra_credit_genereate_8_files()
ameyam_extra_credit_generate_results()
"""

#ameyam_generate_results()
#ameyam_preprocess('/home1/c/cis530/hw3/test_data', 'xml_test')
"""
top_words = extract_top_words('xml_test')
similarity_matrix = extract_similarity(top_words)
top_dependencies =  extract_top_dependencies('xml_test')
prod_rules = extract_prod_rules('xml_test')
process_corpus('xml_test', top_words, similarity_matrix, top_dependencies, prod_rules)
process_corpus('xml_train', top_words, similarity_matrix, top_dependencies, prod_rules)
ameyam_generate_40_files()
"""
#ameyam_generate_results()
#ameyam_generate_classify_documents_result()
#ameyam_extra_credit_train_word2vec()

#ameyam_extra_credit_generate_classify_documents_result()
