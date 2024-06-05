from transformers import set_seed
import pandas as pd
from utils import call_transformer
from preprocessing import pre_processing
from peft import LoraConfig, TaskType
import os
from credentials import access_token


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = access_token


SEED = 123
set_seed(SEED)

R, LORA_ALPHA, LORA_DROPOUT = 64, 32, 0.1
lora_conf = LoraConfig(
    r=R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.SEQ_CLS,
    target_modules='all-linear'
)

# list of models I have tested yet
MODELS = {
    "BERT": "bert-base-cased",
    "DEBERTA": "microsoft/deberta-base",
    "ERNIE": "nghuyong/ernie-2.0-base-en",
    "DEBERTAv3": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "Bertweet": "vinai/bertweet-base",
    "BERT_LARGE": "bert-large-cased",
    "ERNIE_LARGE": "nghuyong/ernie-2.0-large-en",
    "Bertweet_LARGE": "vinai/bertweet-large",
    "MISTRAL_7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma_2B": 'google/gemma-2b'
}

#retrieve the dataframe
train_csv = pd.read_csv('train.csv', sep = ',', header=0)
test_csv = pd.read_csv('test.csv', sep=',', header=0)

#apply pre-processing
train_dataframe, test_dataframe = pre_processing(train_csv, test_csv,apply_Bertweet_normalization=False)



#call_transformer(MODELS,"BERT",train_dataframe, test_dataframe,training_batch_size=8,eval_batch_size=8,learning_rate=2e-5,num_train_epochs=1,weight_decay=0, lora_config= lora_conf, prediction = True , save = True, access_token = access_token)
# accuracy : 0.839754816112084 accuracy on validation set


#call_transformer(MODELS,"DEBERTA", train_dataframe, test_dataframe, training_batch_size=8, eval_batch_size=8, learning_rate=2e-5, num_train_epochs=10,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)
#accuracy :  0.8359580052493438 accuracy on validation set


#call_transformer(MODELS,"ERNIE",train_dataframe, test_dataframe, training_batch_size=8, eval_batch_size=8,learning_rate=1e-5,num_train_epochs=4,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)
# 0.84251968503937 accuracy on validation set for 6 epochs

#call_transformer(MODELS,"DEBERTAv3",train_dataframe, test_dataframe, training_batch_size=16,eval_batch_size=64,learning_rate=5e-06,num_train_epochs=4,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)
#0.8169877408056042 accuracy on validation set


#call_transformer(MODELS,"Bertweet", train_dataframe, test_dataframe, training_batch_size=128, eval_batch_size=128, learning_rate=2e-5, num_train_epochs=4,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)
#accuracy :  0.8241469816272966 accuracy on validation set

#call_transformer(MODELS,"ERNIE_LARGE",train_dataframe, test_dataframe, training_batch_size=8, eval_batch_size=8,learning_rate=2e-5,num_train_epochs=6,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)

#call_transformer(MODELS,"BERT_LARGE",train_dataframe, test_dataframe, training_batch_size=8,eval_batch_size=8,learning_rate=2e-5,num_train_epochs=4,weight_decay=0.01, lora_config= lora_conf, prediction = False , save = False, access_token = access_token)
#bad accuracy

call_transformer(MODELS,"Bertweet_LARGE", train_dataframe, test_dataframe, training_batch_size=2, eval_batch_size=2, learning_rate=1e-5, num_train_epochs=10,weight_decay=0.00, prediction = False , save = False, access_token = access_token)
#0.8293963254593176 accuracy on validation set

#call_transformer(MODELS,"Bertweet_LARGE", train_dataframe, test_dataframe, training_batch_size=8, eval_batch_size=8, learning_rate=2e-5, num_train_epochs=1,weight_decay=0.01, lora_config= lora_conf, prediction = True , save = False, access_token = access_token)
#checkout point 700 and 800 : 0.86 accuracy on validation set

#call_transformer(MODELS,"MISTRAL_7B", train_dataframe, test_dataframe, training_batch_size=2, eval_batch_size=2, learning_rate=2e-5, num_train_epochs=3,weight_decay=0.01, lora_config= lora_conf, prediction = True , save = False, access_token = access_token)
#84% accuracy

#call_transformer(MODELS,"Gemma_2B", train_dataframe, test_dataframe, training_batch_size=8, eval_batch_size=8, learning_rate=2e-5, num_train_epochs=3,weight_decay=0.01, lora_config= lora_conf, prediction = True , save = False, access_token = access_token)
#85% accuracy
