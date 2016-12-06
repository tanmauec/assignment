
# coding: utf-8

# In[1]:

import pandas as pd
import nltk
from nltk.stem.porter import *
from sklearn.utils import shuffle


# In[2]:

data = pd.read_csv('labelledData.txt', sep=",,,", header = None)
data.columns = ["question", "type"]
#print(data)

#shuffle data
data = shuffle(data)


# In[3]:

#the labelled data file has 5 classes in which a question belongs to, 'what', 'when', 'who',
#'affirmative' 'unknown'.

#We can use input sentence to construct a feature vector.
#But, for that first we need to decide what features to have on data set.

'''
Some of features i can think of-
1. wh-word - check which wh word is present in a question.
2. each of wh-word is somewhat different, so, we need to create additional features to capture differences between 
them.

Learning a model for a dataset is an iterative process, u create a model, test it, improve upon it, retest..so on.
'''


# In[4]:

#Some basic text processing functions we can use later!
def tokenize(question):
    tokens = nltk.word_tokenize(question)
    return tokens
    
def stem(tokens):
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed
    
s = "You are looking lovely today!"
tokens = tokenize(s)


# In[5]:

'''
Let's begin with some basic feature-engineering.    
'''
labels = {'what' : 0,
          'who' : 0,
          'when' : 0,
          'unknown' : 0,
          'affirmation' : 0}
for index in range(len(data)):
    type = data['type'][index].strip()
    labels[type] = labels[type] + 1

print(labels)
#So, we see there is fair representation of each class type!!

'''
We need to train features that capture the DISTINCT trait of each type of question.
To understand and use this structure, pos tags will be handy.
'''

#Let's look at affirmation or yes/no sentences to begin with.
'''
for index in range(len(data)):
    type = data['type'][index].strip()
    if type == 'affirmation':
        #print(data['question'][index])
'''
        
#We can observe that unlike other question types, an affirmation always begins with a BE-verb or an
#auxiliary verb.
#This is a strong feature to identify affirmations.

#WHO -> this question type asks about a person.

#lets look at how many sentences which begin with who, get labelled as other classes.

print("\n\nmisclassified questions containing who")

for index in range(len(data)):
    type = data['type'][index].strip()
    if type != 'who' and 'who ' in data['question'][index].lower():
        print("type({}) question({})".format(data['type'][index], data['question'][index]))
        
#so, there are few such misclassified 'who' questions.

#let's check for 'what' now.

print("\n\nmisclassified questions containing what")

for index in range(len(data)):
    type = data['type'][index].strip()
    if type != 'what' and 'what' in data['question'][index].lower():
        print("type({}) question({})".format(data['type'][index], data['question'][index]))


# In[6]:

#So certainly there are misclassifications, let's construct a an initial model, and we will relook
#at misclassifications/confusion 
#matrix to look at misclassifications.

'''
A question of type 'when' asks the time of occurence of certain event.
Therefore, words such as 'time', and verbs that indicate some event or action, can act as strong features
to detect a when type of question.
'''
#Label - 'when' ,

print("misclassified questions containing when")
for index in range(len(data)):
    type = data['type'][index].strip()
    if type != 'when' and 'when' == data['question'][index].split()[0].lower():#first word is when
        print("type({}) question({})".format(data['type'][index], data['question'][index]))

'''
Looking at the cases where when questions get labelled as what, this seems to happen
when there is no verb except BE-VERB suggesting any action happening.
eg. 
when was hurricane hugo ?
when was `` the great depression '' ? 

BUT, we can't act on this conclusion, since this doesn't always happen!

Following two sentences don't have much difference in pos tags, but yet labelled differently!

> when is boxing day - LABEL(what)
[('when', 'WRB'), ('is', 'VBZ'), ('boxing', 'VBG'), ('day', 'NN'), ('?', '.')]

> when is bastille day - LABEL(when)

[('when', 'WRB'), ('is', 'VBZ'), ('bastille', 'VBN'), ('day', 'NN'), ('?', '.')]

'''


# In[7]:

#Label - 'what'

#What question usually asks for some information, clarify some fact.

#Let's first look at questions that begin with 'what' but are labelled differently.
print("\n\nmisclassified questions containing what")

for index in range(len(data)):
    type = data['type'][index].strip()
    if type != 'what' and 'what' == data['question'][index].split()[0].lower():#first word is when
        print("type({}) question({})".format(data['type'][index], data['question'][index]))
        
'''
Looking at the misclassifications, some things are immediately obvious,

- using what to ask time, amounts to making a when question,

- asking about a person is actually a who question(what actor came to dinner in guess who 's coming to dinner ?)

- both 'time' and 'year' relate to time, so we need some mechanism to identify such synonyms or 
    similar meaning words.

- what question, that try to ask about a certain entity(PROPER NOUN), 
  eg. nicholas cage, apricot computer, richard etc.which might itself be unknown, also get labelled as
  unknown!! Is this assumption/inference well found, let's assign features to find out better!!
'''


# In[8]:

#Label - 'who'

#Who question usually asks about a person.

#Let's first look at questions that begin with 'what' but are labelled differently.
print("\n\nmisclassified questions containing who")

for index in range(len(data)):
    type = data['type'][index].strip()
    if type != 'who' and 'who' == data['question'][index].split()[0].lower():#first word is when
        print("type({}) question({})".format(data['type'][index], data['question'][index]))
'''
Looking at the misclassifications, 
3 sentences are labelled as of 'what' type, but it seems they should have been labelled 'who' only.
So, mostly, question that begin with 'who' word, tend to be who questions only.
So, a feature that we have to include is begin word of sentence.

for index in range(len(data)):
    type = data['type'][index].strip()
    if type == 'who':
        print("type({}) question({})".format(data['type'][index], data['question'][index]))
'''


# In[9]:

#Label - 'affirmation'

'''
An affirmaton is a yes/no question, and it mostly begins with a auxiliary-verb.
Let's see if it has some less-obvious labelled examples.
'''

#So, to classify a question as affirmation, 
#it shouldn't be beginning with one of the wh-words, and
#first word shall be a auxiliary verb.

#Lets look at pos tags for some of affirmative sentences.
def pos_tag(question):
    tokens = tokenize(question)
    tagged_tokens = nltk.pos_tag(tokens)
    print(tagged_tokens)


# In[10]:

#Label - 'unknown'
#A question, that doesn't associate well with any of the other 4 classes, will be classified
#as unknown.


# In[11]:

#Model creation
auxiliary_verbs = ['can', 'could', 'shall', 'should', 'do', 'does', 'did', 'am', 'is', 'are', 'was', 'were',
                   'will', 'would', 'has', 'have', 'had']

personal_pronouns = ['i', 'you', 'she', 'we', 'they', 'there', 'anybody', 'anyone', 'somebody', 'someone']

time_synonyms = ['day', 'month', 'week', 'year', 'time']

included_question_classes = ['who','when','what']
excluded_question_classes = ['how', 'which']

#this function defines a feature question_class - class that a question belongs to
#possible values = {'who', 'what', 'when', 'affirmative', 'excluded'}
def question_class(question, features):
    if 'question_class' in features.keys() and 'question_class' == 'excluded_class':
        return features
    first_word = question.split()[0].strip()
    dictionary = {'question_class':'unknown'}
    
    if first_word in auxiliary_verbs:
        dictionary["question_class"] = "affirmative"
        dictionary['leading_question_word'] = 'affirmative'
    else:
        features['question_class'] = features['leading_question_word']
        return features
    
    return dictionary

#this function finds out, if sentence has a wh-word(what, who, when),
#in case of multiple wh-words, it returns the first one found.
def leading_question_word(question):
    dictionary = {"leading_question_word":'unknown'}
    for word in question.split():
        if word in included_question_classes:
            dictionary["leading_question_word"] = word
            break
        else:
            if word in excluded_question_classes:
                dictionary['leading_question_word'] = 'unknown'
                break
    
    return dictionary

#this method finds and returns first word of question as a feature.
def first_word(question):
    dictionary = {}
    dictionary["first_word"] = question.split()[0]
    return dictionary

#this function finds and returns the first word in question that follows a leading_question_word.
def following_word_feature(question_tokens, features):
    dictionary = {}
    if features["leading_question_word"] != 'unknown':
        index = question_tokens.index(features["leading_question_word"])
        if index < len(question_tokens) - 1:
            next_word = question_tokens[index+1].strip()
            if next_word in time_synonyms:
                features["leading_question_word"] = 'which'
                features["question_class"] = "excluded_class"
    return features

#this function finds and returns the preceding word if any, to the leading_question_word.
def preceding_word_feature(question_tokens, features):
    question_tagged = nltk.pos_tag(question_tokens)
    dictionary = {}
    if features['leading_question_word'] == 'what':
        index = question_tokens.index('what')
        dictionary['preceding_word_pos'] = question_tagged[index-1][1]
    return dictionary

def question_features(question):
    question = question.strip("[ .?]")
    question_tokens = tokenize(question)
    features = {}
    features = {**leading_question_word(question), **features}
    features = {**preceding_word_feature(question_tokens, features), **features}
    features = following_word_feature(question_tokens, features)
    features = question_class(question, features)
    return features

s = "what time is it right now ?"
print(nltk.pos_tag((s).split()))
print(question_features(s))


# In[12]:

#Divide into training and test datasets
feature_sets = [(question_features(row['question']), row['type'].strip()) for (index, row) in data.iterrows()]

#Since the dataset is very small(1483 samples), we will divide in 80:20 ratio in training and test sets. 
train_set, test_set = feature_sets[:1186], feature_sets[1186:]


# In[13]:

#Using a naive bayes classifier.
classifier = nltk.classify.NaiveBayesClassifier.train(train_set)

#Most informative features.
classifier.show_most_informative_features()

#accuracy
print("Accuracy on training set:({})".format(nltk.classify.accuracy(classifier, train_set)))

print("Accuracy on test set:({})".format(nltk.classify.accuracy(classifier, test_set)))


'''
With our initial model, we got a 96.6% accuracy on test_set, 
After repeated iteration, we have got a accuracy of 98% on training set, 
but an accuracy of 99% on test set.

To avoid over-fitting, let us perform 10-fold cross-validation on training dataset.
'''
from sklearn import cross_validation
cv = cross_validation.KFold(len(train_set), n_folds=10, shuffle=False, random_state=None)

for traincv, evalcv in cv:
    classifier = nltk.NaiveBayesClassifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
    print('accuracy:({})'.format(nltk.classify.accuracy(classifier, train_set[evalcv[0]:evalcv[len(evalcv)-1]])))
    
#after cross-validation on training set, prediction accuracy
print("Accuracy on training set:({})".format(nltk.classify.accuracy(classifier, train_set)))
print("Accuracy on test set:({})".format(nltk.classify.accuracy(classifier, test_set)))


# In[14]:

#Using a Decision tree classifier.
classifier = nltk.classify.DecisionTreeClassifier.train(train_set)

#Most informative features.
#classifier.show_most_informative_features()

#accuracy
print("Accuracy on training set:({})".format(nltk.classify.accuracy(classifier, train_set)))

print("Accuracy on test set:({})".format(nltk.classify.accuracy(classifier, test_set)))

print(classifier.classify(question_features('hazmat stands for what?')))

'''
With our initial model, we got a 96.6% accuracy on test_set, 
After repeated iteration, we have got a accuracy of 98% on training set, 
but an accuracy of 99% on test set.

To avoid over-fitting, let us perform 10-fold cross-validation on training dataset.
'''
from sklearn import cross_validation

cv = cross_validation.KFold(len(train_set), n_folds=10, shuffle=False, random_state=None)

for traincv, evalcv in cv:
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
    print('accuracy:({})'.format(nltk.classify.accuracy(classifier, train_set[evalcv[0]:evalcv[len(evalcv)-1]])))
    
#after cross-validation on training set, prediction accuracy
print("After 10-fold cross-validation")
print("Accuracy on training set:({})".format(nltk.classify.accuracy(classifier, train_set)))
print("Accuracy on test set:({})".format(nltk.classify.accuracy(classifier, test_set)))


# In[15]:

#Using a Maxent classifier.
classifier = nltk.classify.MaxentClassifier.train(train_set, 'GIS', trace=0, max_iter=100)

#Most informative features.
classifier.show_most_informative_features()

#accuracy
print(nltk.classify.accuracy(classifier, train_set))

print(nltk.classify.accuracy(classifier, test_set))


# In[18]:

#Let's have a quick look at the errors made in classification
errors = []
for i, row in data.iterrows():
    type = row['type'].strip()
    question = row['question'].strip()
    guess = classifier.classify(question_features(question))
    if guess != type:
        errors.append( (type, guess, question) )

print('No of errors are:({})'.format(len(errors)))
for (type, guess, question) in sorted(errors):
    print('actual={:<8} predicted={:<8s} question={:<100}'.format(type, guess, question))




# In[17]:

#Conclusion
'''
There are a total of 30 misclassified examples from the complete dataset,

Some of these appear to be wrongly labelled and hence will also be misclassified by the learning algo,
listing 25 such below:

actual=unknown?  predicted=what     question=what did richard feynman say upon hearing he would receive the nobel prize in physics ?             
actual=unknown?  predicted=what     question=what does a nihilist believe in ?                                                                   
actual=unknown?  predicted=what     question=what game 's board shows the territories of irkutsk , yakutsk and kamchatka ?                       
actual=unknown?  predicted=what     question=what is the name of the managing director of apricot computer ?                                     
actual=unknown?  predicted=what     question=what is the occupation of nicholas cage ?                                                           
actual=unknown?  predicted=what     question=what lawyer won the largest divorce settlement , $85 million , in u.s. history for sheika dena al-farri ?
actual=unknown?  predicted=when     question=when did the berlin wall go up ?                                                                    
actual=unknown?  predicted=when     question=when did the bounty mutiny take place ?                                                             
actual=unknown?  predicted=when     question=when was the first wall street journal published ?                                                  
actual=unknown?  predicted=who      question=name the ranger who was always after yogi bear . ?                                                  
actual=unknown?  predicted=who      question=who is the man behind the pig-the man who pulls the strings and speaks for miss piggy ?             
actual=what??     predicted=when     question=what year did germany sign its nonaggression pact with the soviet union ?                           
actual=what??     predicted=when     question=what year did jack nicklaus join the professional golfers association tour ?                        
actual=what??     predicted=when     question=what year did the united states pass the copyright law ?                                            
actual=what??     predicted=when     question=what year did the vietnam war end ?                                                                 
actual=what??     predicted=when     question=when did rococo painting and architecture flourish ?                                                
actual=what??     predicted=when     question=when is boxing day ?                                                                                
actual=what??     predicted=when     question=when is the tulip festival in michigan ?                                                            
actual=what??     predicted=when     question=when was `` the great depression '' ?                                                               
actual=what??     predicted=when     question=when was hurricane hugo ?                                                                           
actual=what??     predicted=who      question=who portrayed portly criminologist carl hyatt on checkmate ?                                        
actual=what??     predicted=who      question=who was the first host of person to person ?                                                        
actual=what??     predicted=who      question=who was the star of the 1965 broadway hit golden boy ? 


CONCLUSION: All 3 models Naive Bayes, DecisionTree, Maxent perform similar on test set,
and Naiye Bayes gives best accuracy of 98%
'''

