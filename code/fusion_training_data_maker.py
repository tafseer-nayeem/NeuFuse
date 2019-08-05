import sys
import os
import shutil
import errno
import numpy as np
import nltk
import string
import re
import codecs
from sklearn.metrics import jaccard_similarity_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

reload(sys)
sys.setdefaultencoding('utf8')


lowest_threshold = 3.0 #Lowest taken threshold for relatedness score. If the related score is greater than this threshold we will consider the similarity between Sentence A & B
similarity_threshold = 0.25 #To find at least one common word between two sentences
MAX_INPUT_SENTENCE = 5 #There will be MAX_INPUT_SENTENCE sentences in the input pair
sick_file = 'SICK/SICK.txt' #SICK file location
MAX_JACCARD = 0.0
MIN_JACCARD = 1.0
#cnn = "cnn/stories/" #cnn story files location
cnn = "cnn/stories/"


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
puncset = set(string.punctuation)
puncset.remove('-')
SRCDIR = os.path.dirname(os.path.realpath(__file__))
'''
if os.path.exists('lineNumber.txt') :
	os.remove('lineNumber.txt')
if os.path.exists('all_train.txt') :
	os.remove('all_train.txt')
'''

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


def stem_tokens(tokens):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenizer1(text):
	text = re.sub('[^a-zA-Z]', ' ', text)
	tokens = text.lower().split()
	tokens = [lemmatizer.lemmatize(tkn, pos='n') for tkn in tokens]
	tokens = [lemmatizer.lemmatize(tkn, pos='v') for tkn in tokens]
	return tokens


def tokenizer2(text):
	text = "".join([ch for ch in text if ch not in string.punctuation])
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens)
	return stems

def tokenizer3(text):
	tokens = nltk.word_tokenize(text)
	tokens = [i for i in tokens if i not in string.punctuation]
	stems = stem_tokens(tokens)
	return stems

def tokenizer4(text):
	''' Returns a bag of words for the sentence '''
	sentenceWords = re.findall(r"[\w']+", text)
	cleanWords = []
	for word in sentenceWords:
		if word not in SMARTSTOPWORDS:
			cleanWords.append(word)
	return set(cleanWords)

def tokenizer5(doc):
	doc = doc.lower()  # Lower the text.
	doc = nltk.word_tokenize(doc.decode('unicode_escape').encode('ascii','ignore'))  # Split into words.
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

def tokenizer7(sentence):
	sentence = "".join([ch for ch in sentence if ch not in puncset])
	sentence = sentence.lower()
	sentence = sentence.split()
	sentence = [w for w in sentence if not w in SMARTSTOPWORDS]
	sentence = [lemmatizer.lemmatize(w, pos='n') for w in sentence]
	sentence = [lemmatizer.lemmatize(w, pos='v') for w in sentence]
	#sentence = [w for w in sentence if w.isalpha()]  # Remove numbers and punctuation.
	return sentence

def letter_only_function(sentence):
	  #Remove non-letters - Utilized Regex Library
	  letters_only = re.sub("[^a-zA-Z]", " ", sentence)
	  letters_only = letters_only.lower()
	  return letters_only

def jaccard_similarity(string1, string2):
	intersection = set(string1).intersection(set(string2))
	union = set(string1).union(set(string2))
	if len(union) == 0:
		return 0.0
	return len(intersection)/float(len(union))

def similarity(s1, s2):       #Finding out maximum similarity from tokenized combination of tokenizer 5,6 & 7     
	sentence1 = tokenizer5(s1)
	sentence2 = tokenizer5(s2)
	temp_similarity = jaccard_similarity(sentence1, sentence2)
	'''
	print sentence1
	print sentence2
	print temp_similarity
	#raw_input("tokenizer5") #For Python 2
	'''
	sentence1 = tokenizer6(s1)
	sentence2 = tokenizer6(s2)
	temp = jaccard_similarity(sentence1, sentence2)
	if temp > temp_similarity :
		temp_similarity = temp
	'''
	print sentence1
	print sentence2
	print temp_similarity
	#raw_input("tokenizer6") #For Python 2
	'''
	sentence1 = tokenizer7(s1)
	sentence2 = tokenizer7(s2)
	temp = jaccard_similarity(sentence1, sentence2)
	if temp > temp_similarity :
		temp_similarity = temp
	'''
	print sentence1
	print sentence2
	print temp_similarity
	#raw_input("tokenizer7")
	'''
	#if temp_similarity < 0.1 and temp_similarity >0.01 :
		#raw_input()
	return temp_similarity



def read_sick():
	with open(sick_file) as fp:  
		minimum_similarity = 1.0 #assuming minimum similarity between two sentences
		line = fp.readline()
		cnt = 0
		while line:
			#print "Line {}: {}".format(cnt, line.strip())
			line = fp.readline()
			splitted_line = line.split("\t")
			if len(splitted_line) == 1:
				break
			
			#print splitted_line[4]
			if float(splitted_line[4]) > lowest_threshold:
				temp_similarity = similarity(letter_only_function(splitted_line[1]), letter_only_function(splitted_line[2]))

				if temp_similarity < similarity_threshold :
					continue
				elif temp_similarity < minimum_similarity:
					print temp_similarity
					minimum_similarity = temp_similarity
			print "End of Processing Line :: %d" %(cnt)

			cnt += 1
			#if cnt == 10:
			#    break
		print "\n>>>>>>>>>>>>>>>>END OF SICK DATASET READING<<<<<<<<<<<<<<<<\n"
		print "For minimum relatedness score of :: %f" %(lowest_threshold)
		print "Minimum similarity found in Sick DATASET :: %f" %(minimum_similarity)
		raw_input("Press Enter to continue...") #For Python 2
		#input("Press Enter to continue...") #For Python 3        
		return minimum_similarity

def choose_top_sentences(test_url):
	with open(test_url) as story_file:
		flag = 0
		highlight = []
		for line in story_file:
			line = line.rstrip("\n\r")
			if flag == 1:
				highlight=line
			if line == "@highlight":
				flag = 1
	sentences = []
	with open(test_url) as story_file:
		for line in story_file:
			temp_line = line.rstrip("\n\r")
			if temp_line == "@highlight":
				break
			sentences.append(line)
	os.remove(test_url)

	sen_similarity = np.zeros(len(sentences))
	flag_sen = np.zeros(len(sentences))
	i = 0
	for s in sentences:
		sen_similarity[i] = similarity(s, highlight)
		i += 1
	sorted_sim = np.sort(sen_similarity)[::-1]
	#print sen_similarity
	#print sorted_sim
	sen_counter = 0
	for i in range(0,len(sorted_sim)):
		if sen_counter == MAX_INPUT_SENTENCE:
			break
		for j in range(0,len(sen_similarity)):
			if sorted_sim[i] == sen_similarity[j] and sen_similarity[j] != 0:
				flag_sen[j]=1
				sen_counter += 1
			
	file = open(test_url, "w")
	for i in range(0,len(flag_sen)):
		if  flag_sen[i] == 1:
			file.write(sentences[i])
	file.write("\n")
	file.write("@highlight\n\n")
	file.write(highlight)

def make_from_cnn():
	in_folder = "input" #input file directory
	out_folder = "output" #output file directory
	max_input_line = 0
	number_of_sample = 0
	number_of_highlight = 0
	lines = []
	
	try:
		os.makedirs(in_folder)
		os.makedirs(out_folder)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
		else :
			inString = raw_input("There is already folder named 'input' or 'output'!!!\n If you proceed your existing input/output directory will be deleted. \n If you want to proceed enter 'Y' or enter 'N'...")
			if inString == 'Y' or inString == 'y':    
				shutil.rmtree(in_folder, ignore_errors=False, onerror=None)
				shutil.rmtree(out_folder, ignore_errors=False, onerror=None)			
				os.makedirs(in_folder)
				os.makedirs(out_folder)
			else :
				sys.exit('Try Again!!')
	threshold = read_sick()
	all_files = os.listdir(cnn)
	try :
		os.remove("lineNumber.txt")
		os.remove("all_train.txt")
	except :
		pass
	line_count_file = open("lineNumber.txt", "a")
	source_url_list = open("all_train.txt", "a")
	for story in all_files:
		#print "Processing :: %s" %(story)
		story_url = story
		story = cnn + story
		with open(story) as story_file:
			count = 0
			temp_count = -1
			highlight = []
			for line in story_file:
				line = line.rstrip("\n\r")
				count = count + 1
				if line == "@highlight":
					temp_count = count + 2
				if count == temp_count:
					temp_count = -1
					highlight.append(line)
					#print line
			#print highlight
			#print count

			cnt = 0
			number_of_highlight = number_of_highlight + len(highlight)

			#lemmatize noun verb 
			#stop word bad dibo

			#input_url = cnn + "input/" + str(cnt) + ".txt"
			

			for reference in highlight:
				
				input_line_count = 0
				temp_reference = reference
				output_url = out_folder + "/" + story_url + str(cnt) + ".story"
				input_url = in_folder + "/" + story_url + str(cnt) + ".story" 
				file_name =  story_url + str(cnt) + ".story"

				
				with open(story) as story_file:
					for line in story_file:
						reference = letter_only_function(reference)
						line = line.rstrip("\n\r")
						if line == "@highlight":
							break
						try :
							tokenized_sentences = nltk.tokenize.sent_tokenize(line)
						except:
							continue
						
						for sen in tokenized_sentences:
							letter_only_sen = letter_only_function(sen)
							
							try:
								sen_similarity = similarity(reference, letter_only_sen)
							except:
								continue
							
							if sen_similarity > 0.0 and sen_similarity < 1.0:
								if sen_similarity >= MAX_JACCARD :
									global MAX_JACCARD
									MAX_JACCARD = sen_similarity
								if sen_similarity <= MIN_JACCARD :
									global MIN_JACCARD
									MIN_JACCARD = sen_similarity
							
							if sen_similarity < similarity_threshold :
								continue
							if len(letter_only_sen) < len(reference) :
								continue
							elif sen_similarity > threshold:
								
								temp_file = open(input_url, "a")
								#letter_only_sen = letter_only_sen + " @len" + str(len(letter_only_sen.split())) + "\n" #saves file with length.
								sen = sen + '\n' #saves file without length
								lines.append(sen)
								input_line_count += 1
								
								#IF you want to save into file while checking each line
								
								#temp_file = open(input_url, "a")
								temp_file.write(sen)
								temp_file.close()
							
							#print input_url
							#print reference + "\n" + line

							#print similarity(reference, letter_only_sen)
							#print "+++++++++++++++++++\n"
						
						line = letter_only_function(line)
							
						
						'''
						try :
							temp_similarity = similarity(reference, line)
						except :
							continue

						if temp_similarity < similarity_threshold :
							continue
						elif temp_similarity > threshold:
							line = line + " @len" + str(len(line.split())) + "\n" #saves file with length.
							#line = line + "\n" #saves file without length
							lines.append(line)
							input_line_count += 1
							
							#IF you want to save into file while checking each line
							input_url = in_folder + "/" + story_url + ".in" + str(cnt)
							temp_file = open(input_url, "a")
							temp_file.write(line)
							temp_file.close()
							
							#print input_url
							#print reference + "\n" + line
						'''
				'''

				input_url = in_folder + "/" + story_url + ".in" + str(cnt)
				temp_file = open(input_url, "w")
				temp_file.write(lines)
				temp_file.close()
				'''

				line_count_file.write(str(input_line_count)+" "+input_url+" "+output_url+"\n")

				
				#temp_file.write(reference) #saves without length
				if os.path.exists(input_url) and os.path.getsize(input_url) > 0:
					#print "size 0000000000000000000000000000000"
					#temp_file = open(output_url, "w")
					temp_file = open(input_url, "a")
					temp_reference = "\n@highlight\n\n" + temp_reference
					#temp_file.write(reference + " @len" + str(len(reference.split()))) #saves with length
					temp_file.write(temp_reference)
					temp_file.close()

					choose_top_sentences(input_url) #Comment this line, if you want all the sentences in the source

					input_url = cnn + file_name + "\n"
					source_url_list.write("tmp_dir/"+input_url)
					
					
					number_of_sample += 1
					sys.stdout.write("\r\x1b[KNumber of Sample Created:"+str(number_of_sample).__str__()+" ::: Number of Highlights:"+str(number_of_highlight).__str__())
					sys.stdout.flush()
				
				if input_line_count > max_input_line :
					max_input_line = input_line_count
					max_line_file_name = input_url
				input_line_count = 0
				cnt = cnt + 1
	line_count_file.close()
	source_url_list.close()
	print "\n\n"
	print max_input_line
	print max_line_file_name


def main():
	make_from_cnn()
	print "Maximum Similarity Found in DATASET :: " + str(MAX_JACCARD)
	print "Minimum Similarity Found in DATASET :: " + str(MIN_JACCARD)

if __name__ == "__main__":
	main()
