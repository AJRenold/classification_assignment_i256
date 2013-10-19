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
