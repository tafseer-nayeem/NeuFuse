from glob import glob
import sys
import os
import shutil
import errno
import argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import numpy as np
import nltk
import string
from sklearn.metrics import jaccard_similarity_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from keras.layers import Input, Embedding, Flatten, Reshape
from keras.layers import Dense, Conv1D, Dropout, merge
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, ZeroPadding1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from utils import cluster_quality
import keras
#reload(sys)
#sys.setdefaultencoding('utf-8')
output = True
dim = 300
N_EPOCH = 50
N_CLUSTERS = 20 #8/20
N_SAMPLES = 3000
# N_CLUSTERS = 8
# EMBEDDING_FILE = 'data/glove_model2.txt'
EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin'
# EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin'
TEXT = []

#test_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
#test_model.save_word2vec_format('data/GoogleNews-vectors-negative300.txt', binary=False)

# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False) #google e true
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True) #google e true


parser = argparse.ArgumentParser(description='Deep Recurrent Generative Decoder for Abstractive Text Summarization in DyNet')

parser.add_argument('--type', type=str, default='single', help='single/multi  \
	single :: only one doc file where every sentence is separated by a new line. Dont forget to set file_location to your directory  \
	multi  :: folder name where multi documents data are saved. Format : DUC2004 \
	[default: single]')
# parser.add_argument('--file_location', type=str, default='search_snippet_input.txt', help='Location to your file [default: StackOverflow.txt]')
parser.add_argument('--file_location', type=str, default='data/StackOverflow.txt', help='Location to your file [default: StackOverflow.txt]')

args = parser.parse_args()
print(args)




print "  "
print "  "
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
puncset = set(string.punctuation)
puncset.remove('-')
SRCDIR = os.path.dirname(os.path.realpath(__file__))
def getEnglishStopWords():
	'''
	returns a set of stop words for NLP pre-processing
	from nltk.corpus.stopwords()
	'''
	stop_words = set(stopwords.words("english"))
	   
	return stop_words


def _build_stop_words_set():
	'''
	Build set of stop words to ignore.
	'''

	# source: http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
	return set(open(os.path.join(SRCDIR,'smartstop.txt'), 'r').read().splitlines())

SMARTSTOPWORDS = _build_stop_words_set()

def tokenizer5(doc):
	doc = doc.lower()  # Lower the text.
	doc = nltk.word_tokenize(doc)  # Split into words.
	doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
	doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
	return doc

def tokenizer6(sentence):
	sentence = sentence.translate(None, string.punctuation)
	sentence = sentence.lower()
	sentence = sentence.split()
	sentence = [w for w in sentence if not w in SMARTSTOPWORDS]
	sentence = [lemmatizer.lemmatize(w, pos='n') for w in sentence]
	sentence = [lemmatizer.lemmatize(w, pos='v') for w in sentence]
	sentence = [w for w in sentence if w.isalpha()]  # Remove numbers and punctuation.
	return sentence

def binarize(target):
	median = np.median(target, axis=1)[:, None]
	binary = np.zeros(shape=np.shape(target))
	binary[target > median] = 1
	return binary




def get_model(embedding_matrix, nb_words, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, TARGET_DIM):
	embedding_matrix_copy = embedding_matrix.copy()
	trainable_embedding = False
	# Embedding layer
	pretrained_embedding_layer = Embedding(
		input_dim=nb_words,
		output_dim=EMBEDDING_DIM,
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
	)

	# Input
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = pretrained_embedding_layer(sequence_input)
	
	# 1st Layer
	#x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
	#x = GlobalMaxPooling1D()(x)
	x = keras.layers.GRU(100, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False) (embedded_sequences)

	# Output
	x = Dropout(0.02)(x)
	predictions = Dense(TARGET_DIM, activation='softmax')(x)
	model = Model(sequence_input, predictions)

	model.layers[1].trainable=trainable_embedding

	adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	# Loss and Optimizer
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['mae'])
	# Fine-tune embeddings or not
	model.summary()
	return model

def multi2single():
	subd = [s.rstrip("/") for s in glob("*/")]
	save_folder = args.type
	try:
		os.makedirs(save_folder)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
		else :
			inString = raw_input("There is already folder named 'combined'!!!\n If you proceed your existing directory will be deleted. \n If you want to proceed enter 'Y' or enter 'N'...")
			if inString == 'Y' or inString == 'y':    
				shutil.rmtree(save_folder, ignore_errors=False, onerror=None)
				os.makedirs(save_folder)
			else :
				sys.exit('Try Again!!')

	for dir in subd :
		all_files = os.listdir(dir+"/")
		if os.path.exists(save_folder+"/"+dir+'.txt') :
			os.remove(save_folder+"/"+dir+'.txt')
		if dir != "combined":
			file_writer = open(save_folder+"/"+dir+".txt","a")
			for story in all_files:
				with open(dir+"/"+story) as story_file:
					for line in story_file:
						file_writer.write(line)

	file_writer.close()
	return save_folder

def final_cluster(merged_file):
		
	tempdata = [text.strip() for text in merged_file]
	#data = tokenizer5(data)
	tmp = []
	for i in tempdata :
		tmp.append(i)
	data = []
	for line in tempdata:
		data.append(' '.join(tokenizer5(line)))
	#print data
	global TEXT
	TEXT = tmp
	print("Total: %s texts" % format(len(data), ","))

	tokenizer = Tokenizer(char_level=False)
	tokenizer.fit_on_texts(data)
	sequences_full = tokenizer.texts_to_sequences(data)
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	MAX_NB_WORDS = len(word_index)
	seq_lens = [len(s) for s in sequences_full]
	print("Average length: %d" % np.mean(seq_lens))
	print("Max length: %d" % max(seq_lens))
	MAX_SEQUENCE_LENGTH = max(seq_lens)

	X = pad_sequences(sequences_full, maxlen=MAX_SEQUENCE_LENGTH)		
		
	print('Preparing embedding matrix')
	#exit()
	#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
	EMBEDDING_DIM = dim
	nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
	embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
	for word, i in word_index.items():
		if word in word2vec.vocab:
			embedding_matrix[i] = word2vec.word_vec(word)
		else:
			print(word)
	print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

	#for line in merged_file:
		#print line


		
	Y = {}
	tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
	denom = 1 + np.sum(tfidf, axis=1)[:, None]
	normed_tfidf = tfidf/denom
	average_embeddings = np.dot(normed_tfidf, embedding_matrix)

	Y["ae"] = average_embeddings
	print("Shape of average embedding: ", Y['ae'].shape)

	# binary Y
		
	reduction_name = "ae"
	B = binarize(Y[reduction_name])

	# Last dimension in the CNN
	TARGET_DIM = B.shape[1]

	# Example of binarized target vector
	print(B.shape)
	print(B[0])
	print(TARGET_DIM)
	'''
	embedding_matrix
	nb_words
	EMBEDDING_DIM
	MAX_SEQUENCE_LENGTH
	'''
	
	nb_epoch = N_EPOCH
	checkpoint = ModelCheckpoint('models/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	model = get_model(embedding_matrix, nb_words, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, TARGET_DIM)
	model.fit(X, B, validation_split=0.2,epochs=nb_epoch, batch_size=100, verbose=1, shuffle=True)      
		
	input = model.layers[0].input
	output = model.layers[-2].output
	model_penultimate = Model(input, output)

	H = model_penultimate.predict(X)
	print("Sample shape: {}".format(H.shape))


	'''
	##############################
	#Try Agglomerative Clustering#
	##############################
		
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenizer5, ngram_range=(1,3))
	tfidf_matrix = tfidf_vectorizer.fit_transform(data)
	terms = tfidf_vectorizer.get_feature_names()
	print terms
	print tfidf_matrix.shape

	dist = 1 - cosine_similarity(tfidf_matrix)

	print dist.shape
		
	ward = AgglomerativeClustering(n_clusters=10, linkage='average').fit(dist)
	pred = ward.labels_

	'''

		
	
	
	# shutil.rmtree("result.txt", ignore_errors=True, onerror=None)
	# result_file = open("result.txt","a+")
	# for i in pred:
	# 	result_file.write(str(i)+"\n")
	# model.save_weights("model.plk")
	label_path = 'StackOverflow_gnd.txt'
	# label_path = 'search_snippet_output.txt'
	with open(label_path) as f:
		target = f.readlines()
	target = [int(label.rstrip('\n')) for label in target]
	y = target
	true_labels = y
	n_clusters = len(np.unique(y))

	#K-Means
	km = AgglomerativeClustering(n_clusters=N_CLUSTERS)
	result = dict()
	V = normalize(H, norm='l2')
	km.fit(V)
	pred_km = km.labels_
	# a = {'deep': cluster_quality(true_labels, pred_km)}


	return pred_km#print(pred)
		





if args.type == 'multi':
	save_folder = multi2single()
	all_files = os.listdir(save_folder+"/")
	for files in all_files:
		with open(save_folder+"/"+files) as merged_file:
			print ("!-------------------------------Processing file ::: "+ files + "-------------------------------!")
			pred = final_cluster(merged_file)
if args.type == 'single':
	with open(args.file_location) as merged_file:
		print ("!-------------------------------Processing file ::: "+ args.file_location + "-------------------------------!")
		pred = final_cluster(merged_file)

if args.type == 'test':
	model = load_model('model.plk')
	print 'ok'

if output:
	label_path = 'data/StackOverflow_gnd.txt'
	with open(label_path) as f:
		target = f.readlines()
	target = [int(label.rstrip('\n')) for label in target]
	y = target
	true_labels = y
	n_clusters = len(np.unique(y))
	
	# print("Number of classes: %d" % n_clusters)
	# km = KMeans(n_clusters=n_clusters, n_jobs=10)
	# result = dict()
	# V = normalize(H, norm='l2')
	# km.fit(V)
	# pred = km.labels_
	
	#print(pred)
	a = {'deep': cluster_quality(true_labels, pred)}


fileWrite = open('cluster_out.txt','a+')
for i in pred:
	fileWrite.write(str(i)+'\n')

