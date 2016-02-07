import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def binary_vector(finalFeaturesList, data1):
    vector_len = len(finalFeaturesList)
    new_feature = []
    list_vectors = []
    for itemList1 in data1:
        new_feature = [0] * vector_len
        for word in itemList1:
            if word in finalFeaturesList:
                pos = finalFeaturesList.index(word)
                new_feature[pos] = 1
        list_vectors.append(new_feature)
    return list_vectors

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    #Milestone 1: Data should not contain stop words, and we should consider unique words only
    stopwords = set(nltk.corpus.stopwords.words('english'))
    featurePosFinalSet = []
    featureNegFinalSet = []

    for itemList in train_pos:
    	featurePosFinalSet.extend(list(set(itemList) - (stopwords)))

    for itemList1 in train_neg:
    	featureNegFinalSet.extend(list(set(itemList1) - (stopwords)))

    postiveWordWithFreq ={}
    negativeWordWithFreq = {}

    for w in featurePosFinalSet:
    	postiveWordWithFreq[w] = postiveWordWithFreq.get(w,0)+1

    for w1 in featureNegFinalSet:
    	negativeWordWithFreq[w1] = negativeWordWithFreq.get(w1,0)+1

    len_pos = len(train_pos)
    len_neg = len(train_neg)

    finalFeaturePostiveList = []
    finalFeatureNegativeList = []

    for key in postiveWordWithFreq.keys():
    	if(float(postiveWordWithFreq[key]/float(len_pos) >= 0.01)):
    		negative_value = negativeWordWithFreq[key]
    		if postiveWordWithFreq[key] >= 2*negative_value:
    			finalFeaturePostiveList.append(key)

    for key in negativeWordWithFreq.keys():
    	if(float(negativeWordWithFreq[key]/float(len_neg) >= 0.01)):
    		positive_value = postiveWordWithFreq[key]
    		if negativeWordWithFreq[key] >= 2*positive_value:
    			finalFeatureNegativeList.append(key)

    #print finalFeatureNegativeList
    #print finalFeaturePostiveList
    finalFeaturesLists = list(set(finalFeaturePostiveList)) + list(set(finalFeatureNegativeList)-set(finalFeaturePostiveList))
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = binary_vector(finalFeaturesLists,train_pos)
    train_neg_vec = binary_vector(finalFeaturesLists,train_neg)
    test_pos_vec = binary_vector(finalFeaturesLists,test_pos)
    test_neg_vec = binary_vector(finalFeaturesLists,test_neg)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def createLabeledSentence(datasets, label):
	labeledSentObjList = []
	count  = 0
	for line in datasets:
		labeledSentObjList.append(LabeledSentence(line, [label+'%s' % count]))
		count = count + 1
	#print labeledSentObjList
	return labeledSentObjList

def createDocVector(datasets,label,model):
	featureVector = []
	count = 0
	for line in datasets:
		featureVector.append(model.docvecs[label+'%s' % count])
		count = count + 1
	return featureVector

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = createLabeledSentence(train_pos, "TRAIN_POS_")
    labeled_train_neg = createLabeledSentence(train_neg, "TRAIN_NEG_")
    labeled_test_pos =  createLabeledSentence(test_pos, "TEST_POS_")
    labeled_test_neg =  createLabeledSentence(test_neg, "TEST_NEG_")

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    """
    http://stackoverflow.com/questions/31321209/doc2vec-how-to-get-document-vectors
    """
    # YOUR CODE HERE
    train_pos_vec = createDocVector(train_pos,"TRAIN_POS_",model)
    train_neg_vec = createDocVector(train_neg,"TRAIN_NEG_",model)
    test_pos_vec = createDocVector(test_pos,"TEST_POS_",model)
    test_neg_vec = createDocVector(test_neg,"TEST_NEG_",model)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["p"]*len(train_pos_vec) + ["n"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    model1 = BernoulliNB(alpha=1.0, binarize=None)
    X = train_pos_vec + train_neg_vec
    nb_model = model1.fit(X,Y)

    model = LogisticRegression()
    lr_model = model.fit(X,Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["p"]*len(train_pos_vec) + ["n"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    model = GaussianNB()
    #print train_pos_vec[0]
    nb_model = model.fit(train_pos_vec+train_neg_vec, Y)
    
    model1 = LogisticRegression()
    lr_model = model1.fit(train_pos_vec+train_neg_vec, Y)

    return nb_model, lr_model

def return_stat(predictions, assignedLabel):
	count  = 0
	for predict in predictions:
		if predict == assignedLabel:
			count = count + 1
	return count

def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    predictions = model.predict(test_pos_vec)
    fn = return_stat(predictions, 'n')
    tp = len(predictions) - fn

    predictions = model.predict(test_neg_vec) 
    fp = return_stat(predictions, 'p')
    tn = len(predictions) - fp

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)

    accuracy = float(tp+tn)/float(tp+tn+fp+fn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
