# toxic-comments-classification
1. main.py is the entrance of the total project.
2. word_parse.py is the module which parses the input the model;
  1st: Process the word inputs to the vector arrays. 
  2nd: Get the train and test dataset,
  3rd: Get the embedding matrix from input text and word2vec text or module;
3. model.py is the model of the project, now only naive LSTM & Attention model is implemented.
4. If you want to run the code: 
    1st: You have to download the 'glove.840B.300d.txt' from 'https://nlp.stanford.edu/projects/glove/';
         Put this file into 'embeds/' directory.
    2nd: Download the kaggle dataset from 'https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data';
         Put these files into 'data/' directory.
5. After you run the code, a submission file will be derived in 'submission_file/'.
6. More to be implemented ......
