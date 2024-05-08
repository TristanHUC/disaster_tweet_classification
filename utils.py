
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import accuracy_score, precision_score , recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, BitsAndBytesConfig
import pandas as pd

def prepare_data(train_dataframe, tokenizer, dataset_class, *, shuffle=False, max_length=None, prediction = False):

    if prediction:
        list_train_text = train_dataframe.iloc[:,3].tolist()
        list_train_label = train_dataframe.iloc[:,4].tolist()

        # tokenize text
        train_encodings = tokenizer(list_train_text, truncation=True)

        # create dataset
        train_dataset = dataset_class(train_encodings, list_train_label)
        validation_dataset = train_dataset

    #not prediction mode => need validation dataset
    else :

        #split the data
        train_text, valid_text, train_labels, valid_labels = train_test_split(train_dataframe.iloc[:,3], train_dataframe.iloc[:,4], test_size = 0.1, shuffle= False)

        list_train_text = train_text.tolist()
        list_train_label = train_labels.tolist()
        list_valid_text = valid_text.tolist()
        list_valid_label = valid_labels.tolist()

        # tokenize text
        train_encodings = tokenizer(list_train_text, truncation=True)
        valid_encodings = tokenizer(list_valid_text, truncation=True)

        # create dataset
        train_dataset = dataset_class(train_encodings, list_train_label)
        validation_dataset = dataset_class(valid_encodings, list_valid_label)

    return validation_dataset, train_dataset


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    # F1 score calcul
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    F1_score = 2 * (precision * recall) / (precision + recall)
    return {"accuracy": acc, 'precision':precision,'recall': recall , "F1_score": F1_score}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.text.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)


def make_prediction(MODELS, model_name, test_dataframe, checkpoint = None, model_passed = None, access_token = None):

    model_source = MODELS[model_name]
    #retrieve the right tokenizer
    if model_name == "MISTRAL_7B":
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=access_token, padding_side="left",
                                                  add_eos_token=True, add_bos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=access_token)

    #if the model is contained in a file :
    if checkpoint:
        #can't load a Bertweet model from huggingface (problem of with the name of weight_layer in the .safetensors file  != name from_pretrained)
        if model_name == "Bertweet_LARGE" or model_name == "Bertweet":
            print("not a model of peft, can't load a Bertweet model from checkpoint")
            print("trying to load a Bertweet model from last adapter_config.json and adapter_model.safetensors files")

            try :
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained('adapter_config.json')
                model_trained = AutoModelForSequenceClassification.from_config(config)
                model_trained.load_state_dict(torch.load('adapter_model.safetensors'))
            except :
                print("loading from adapter_config.json and adapter_model.safetensors has failed")
                print("trying to load from model.pt")

                model_file = 'model_' + model_name + '.pt'
                model_trained = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2,token=access_token)
                model_trained.load_state_dict(torch.load('model.pt'))

        else :
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            if model_name == "MISTRAL_7B":
                model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token, quantization_config=bnb_config)
                model.config.pad_token_id = tokenizer.pad_token_id
                model.gradient_checkpointing_enable()
                model = prepare_model_for_kbit_training(model)
            elif model_name == "Gemma_2B":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token,
                                                                           quantization_config=bnb_config)
                model = prepare_model_for_kbit_training(model)
            else :
                model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token)

            #load the trained weights from the checkpoint directory : exemple 'checkpoint-952'
            model_trained = PeftModel.from_pretrained(model, checkpoint)

    #if the model is passed as argument
    elif model_passed:
        print('model_passed argument')
        model_trained = model_passed
    else :
        print("no checkpoint passed and no model passed, can't make prediction")

    predictions = []
    model_trained.eval()

    clf = pipeline("text-classification", model_trained, tokenizer=tokenizer)

    print('prediction started')
    with (torch.no_grad()):
        for text in test_dataframe.values:
            prediction = clf(text)
            prediction = int(prediction[0]['label'].split('_')[1])
            predictions.append(prediction)
        return predictions



def call_transformer(MODELS, model_name, train_dataframe, test_dataframe, training_batch_size, eval_batch_size, learning_rate,num_train_epochs, weight_decay, lora_config = None, prediction = False , save = False, access_token = None):
# call_transformer : 2 modes :
# prediction = False : Divide the training set in train and validation. Allows to see the training phase and, at this end, compute metric of model with the best accuracy on training set
# prediction = True : Train on 100% of the training set and make a prediction with the model trained on the max epochs. Don't look at the accuracy because training set = validation set*

# Other parameters :
# MODELS : a dictionary to link name of the model to its name in huggingface
# model_name : the name of the model in the dictionary
# Lora_config in order to make the training less ressource consuming and with same performances
# save = True : to save the model (either the best accuracy model or the last : depends on the prediction argument passed)

    model_source = MODELS[model_name]
    if model_name == "MISTRAL_7B":
    #    custom_config = {"pad_token_id": 2} tokenizer = AutoTokenizer.from_pretrained(model_source, token=access_token, padding_side="left", add_eos_token=True,add_bos_token=True, config = custom_config) #**kwarg is when we pass dict in argument and *arg is when we just pass values in argument and it puts it into a list
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=access_token, padding_side="left", add_eos_token=True,add_bos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
    #    custom_config = {}
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=access_token)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    if model_name == "MISTRAL_7B":
        model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token,quantization_config=bnb_config)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)  #convert the weight into int4 or int8 : decrease the weight of the model because the information is stock in 4 or 8 bits instead of 32. Keep the stability of the model because changement in the distribution of values on bits and other optimizations
    elif model_name == "Gemma_2B":
        model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token)
        model = prepare_model_for_kbit_training(model)
    else :
        model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2, token=access_token)

    # max length of 512 for ERNIE as it is not predefined in the model
    if model_name == "ERNIE" or model_name == "ERNIE_LARGE":
        validation_dataset, train_dataset = prepare_data(train_dataframe, tokenizer, Dataset, max_length=512, prediction=prediction)
    else:
        validation_dataset, train_dataset = prepare_data(train_dataframe, tokenizer, Dataset, prediction=prediction)

    if lora_config:
        final_model = get_peft_model(model, lora_config) #not training on all parameters : training is less ressource consuming and has same performance
    else :
        final_model = model

    if model_name == "MISTRAL_7B":
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        final_model = accelerator.prepare_model(final_model)

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,  # total number of training epochs
        learning_rate=learning_rate,
        per_device_train_batch_size=training_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
        weight_decay=weight_decay,  # strength of weight decay
        #save_strategy="steps",  # Save the model checkpoint every logging step
        #save_steps=100,  # Save checkpoints every 100 steps
        #evaluation_strategy="steps",  # Evaluate the model every logging step
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio = 0.06,
        output_dir='.',
        load_best_model_at_end=True)

    trainer = Trainer(model=final_model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=validation_dataset,
                      compute_metrics=compute_metrics,
                      tokenizer=tokenizer)

    #lora_model.print_trainable_parameters()
    trainer.train()


    print('model : ', model_name)

    if prediction:
        pred = make_prediction(MODELS, model_name,test_dataframe,model_passed = final_model, access_token=access_token)
        sample_submission = pd.read_csv("sample_submission.csv")
        sample_submission['target'] = pred
        filename = f'submission_'+model_name+'_'+str(num_train_epochs)+'.csv'
        sample_submission.to_csv(filename, index=False)


    # save model
    if save:
        final_model.save_pretrained('./')
        model_file = 'model_'+model_name+'.pt'
        torch.save(final_model.state_dict(), model_file)

