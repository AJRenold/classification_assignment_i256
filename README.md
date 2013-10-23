Fall 2013 i256 Applied NLP Project 1 - Group Classifier
------

Team Members:

Jeff Tsui - tsui.jeff@gmail.com

AJ Renold - alexander.renold@gmail.com

Siddharth Agrawal - siddharth@ischool.berkeley.edu

Haroon Rasheed - haroonrasheed@berkeley.edu

Code Structure
------------
driver.py - Due to the large number of features we have, driver.py contains the code to generate the model and perform cross validation. It imports features from the files listed at the top, where each feature method returns a dictionary. 

label_test_data.py - code to produce properly formatted output

output_G9.txt - output file for test set

dependencies - you may need to "sudo pip install progressbar". it's a pretty useful tool to vew progress.

The main() method loads the training and held out data and converts them into a list of tuples containing tags and sentences. We sanitize the sentences and convert the tags to 0,-1,1 and strip whitespace and remove non-ascii characters. Each feature method may have helper methods that are needed to compute each feature. For example our adjective feature requires computation of most relevant adjectives that suggest sentiment. After doing preprocessing for features, we loop over each sentence and convert them into features. We played around with different combinations of features (as seen by the large number of methods that are commented out). To evaluate our features, we pass them into the evaluate method which trains models listed in the classifiers variable (naive bayes and decision tree) and peforms k fold cross validation. The resulting average accuracy and confusion matrix is printed.

Features
------------
Adjectives - We extract a few features by finding adjectives that reflect sentiment. One features gets the part of speech tags for each sentence and collects all words tagged 'JJ'. Using a FreqDist, we retrieve the top 20 positive and top 20 negative adjectives, where positive/negative is determined by the tag of the sentence that contains the adjective. The features are whether the sentence contains the word or not. For another feature, we also found a  curated list of positive and negative sentiment words. For this feature we calculate the number of pos and neg words in the sentence that exist in each set. Since the NaiveBayesClassifier only takes a boolean value for each feature, we bucket the values into ranges 0, 1-3, and 4+.

Punctuation Features - Here we implemented the functions suggested in "Gender Classification" paper discussed in class. It covers following features:
1. Syntactic features
 1.1 Exclamation marks - This feature returns the ratio of no. of exclamation marks and sentence
 1.2 Question marks - This feature returns the ratio of no. of question marks and sentence
 1.3 Commas - This feature returns the ratio of no. of commas marks and sentence
 1.4 Semicolons - This feature returns the ratio of no. of semicolons marks and sentence
2. Word based features
 2.1 Uppercase words - This feature returns the count of all uppercase words in the sentence.
 2.2 Length of the review - This feature returns the total length of the input sentence
3. Emoticons - This feature returns the presence of emoticons (':)', ':-)', '=)') in sentence

Bigrams - We extracted the bigram features by selecting the best bigrams based on the information collocation and likelihood ratio. We used nltk's BigramCollocationFinder method to find the top bigrams based on frequency distribution and used likelihood ratio function to score the bigrams. We did this for both positive and negative sentences and extracted the bigram features. We also selected bigram features by finding the Conditional Probability Distribution of the bigrams for both positive and negative sentences and chose the bigrams with high difference between positive and negative probabilities. In both the cases (Bigram collocation and probability distribution) we got similar classification accuracy. We however did not use the bigram features in the final classification with the test dataset as there was not much improvement in classification accuracy by including bigram features.

Unigram Features - We extracted unigram features by choosing N unigrams which maximized class probability between positive and negative sentiment in the training data. We limited this vocabulary by considering only unigrams which appeared more than 15 times in either class and unigrams which were longer than 2 characters. We then reviewed the results of this selection to develop a list of excluded words which were indicative of a product features rather than sentiment.

POS Pattern Bigrams - We used the Part of Speech patterns mentioned in the Turney paper and developed a method for matching those patterns in the tagged training data. We then used the same class probability difference maximization to select the N best phrases from these patterns. We experimented with different methods of feature labelling based on this data, including a count of the presense of positive patterns and negative patterns, and the presence of any of the N patterns in a sentence.


