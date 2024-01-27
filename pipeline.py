import preprocess, postprocess
from utilities import *
from trainer import Trainer
import json
from transformers import TrainingArguments, AutoTokenizer, DistilBertForSequenceClassification
from transformers import TrainerCallback, EarlyStoppingCallback, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer
import pandas as pd


class Pipeline:
    def __init__(self):
        self.model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        with open("conf.json") as f:
            conf = json.load(f)
        os.environ["openai_key"] = conf["openai_key"]

    def run(self, dataset_path, method):
        with open(dataset_path) as f:
            dataset_metadata = json.load(f)

        y_true = []  # true labels
        y_score = []  # predicted scores
        csv_file_path = dataset_metadata["csvFileName"]
        dataset = pd.read_csv(csv_file_path)
        pre_processor = preprocess.Preprocessor(self.tokenizer, self.model)
        question, positiveLabel, negativeLabel = pre_processor.run(dataset, method, dataset_metadata["labelColumnName"])

        if question is None:
            question = dataset_metadata["question"]

        print(f"question: {question}")
        print(f"positive label: {positiveLabel}")
        print(f"negative label: {negativeLabel}")
        test_data = None
        inputSuffix, maxLength = set_dataset_params(question, positiveLabel, negativeLabel)
        tokenized_datasets, test_data = pre_processor.processDataSet(dataset, dataset_metadata['labelColumnName'],
                                                                     positiveLabel, negativeLabel)

        trainer_instance = Trainer(self.tokenizer, self.model)
        trainer = trainer_instance.get_trainer(tokenized_datasets)
        trainer.train()
        trainer.save_model()

        post_processor = postprocess.PostProcessor(self.tokenizer, self.model)
        post_processor.run_test_data(test_data, inputSuffix, dataset_metadata['labelColumnName'])
        post_processor.calculate_statistics()
