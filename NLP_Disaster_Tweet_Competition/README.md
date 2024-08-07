# Natural Language Processing with Disaster Tweets


This project was for a kaggle competition where we are supposed to determine whether or not a twitter user is tweeting about an actual disaster, or a false disaster. This was my first NLP task, it was a fun way to introduce me to it.


# First Model:


For the first model I used chat gpt to teach me about different libraries I could use to preprocess text.


Preprocessing: Text cleaning and normalization.
Feature Extraction: TF-IDF vectorization.
Models: RandomForest, GradientBoosting, and Voting Ensemble.
Evaluation: Validation accuracy and classification report.
Prediction: Generate predictions for the test set and save to CSV.


# Second Model:


For the second model I used BERT (Bidirectional Encoder Representations from Transformers)


Preprocessing: Text normalization and cleaning.
Visualization: Word cloud for disaster-related tweets.
Model: BERT for sequence classification.
Evaluation: Accuracy, precision, recall, F1-score, and ROC AUC score.


# Third Model:


The third model was taken from a discussion. It looks advanced and I spent hours trying to understand it. The tokenization file was using outdated 'unicode' so I had to switch that to str. Overall, I didn't even run this model.



I wrote this while following a tutorial through a book. This was done when I was first learning pytorch.
