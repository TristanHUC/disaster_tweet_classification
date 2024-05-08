---
library_name: peft
base_model: bert-base-cased
---
## Small Project to try to classify tweets related to disasters
The idea comes from Kaggle 2018 NLP challenge : https://www.kaggle.com/competitions/nlp-getting-started/overview 

# References
I have used come from :
- The repository of the paper : "Transformers are Short-text Classifiers" : https://github.com/FKarl/short-text-classification/blob/main
- The article about this kaggle challenge : https://medium.com/@elledled/how-to-reach-the-top-5-in-a-kaggle-nlp-competition-disaster-tweets-c6eccc40bf1a
- The Kaggle notebook : https://www.kaggle.com/code/pardeep19singh/fine-tune-gemma-2b-84-4-on-disaster-tweets/notebook
- The Brev notebook : https://console.brev.dev/notebooks/mistral-finetune-own-data
- The tweetNormalizer functions from bertweet : #reference : https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py

# Repository 
- main.py : contains the code to execute 
- Notebook_format.ipynb : same but as a notebook 
- utils.py : contains all the functions used

# How to run : 
In the main.py or in the notebook, just run the call_transformer function with the parameters you want

# Work in progress : 
- Impossible yet to load a Bertweet model from file using make_predictions function. Therefore, predictions can only be done using call_transformer with prediction = True


### Framework versions

- PEFT 0.10.1.dev0