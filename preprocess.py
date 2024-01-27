from openai import OpenAI
from utilities import *
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from processor import Processor


def get_label_categories(columns, output_column):
    client = OpenAI(api_key=os.environ["openai_key"])

    response_message = []

    while len(response_message) != 3:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"We need to solve a classifcation problem, "
                                            f"We have a database with the following columns {', '.join(columns)}, where {output_column} is the outcome column we want to predict."
                                            "Please generate carefully the best question and labels to use on the column we want to predict .\n"
                                            "IMPORTANT: Give only the question and the labels seperated by comma\n"
                                            "use this format: \n"
                                            "Question, positive_label, negative_label"
                 },
            ]

        )
        response_message = response.choices[0].message.content.replace('\n', '').split(',')

    return response_message[0], response_message[1], response_message[2]


class Preprocessor(Processor):

    def run(self, dataset, method, label_column_name):
        assert method in ["true-false", "yes-no", "llm-based"]
        question = None
        positiveLabel = None
        negativeLabel = None

        if method == "true-false":
            positiveLabel = "true"
            negativeLabel = "false"
        elif method == "yes-no":
            positiveLabel = "yes"
            negativeLabel = "no"
        elif method == "llm-based":
            question, pos_label, neg_label = get_label_categories(list(dataset.columns), label_column_name)
            question = question
            positiveLabel = pos_label
            negativeLabel = neg_label

        return question, positiveLabel, negativeLabel

    def processDataSet(self, data, question, label_column_name, positive_label, negative_label):

        if len(data) > 5000:
            data = data.head(5000)
        train_data = data.sample(frac=0.8, random_state=25)  # 80% for training
        test_data = data.drop(train_data.index)  # 20% for testing

        dataset = DatasetDict(
            train=Dataset.from_dict(
                get_string_data(train_data, question, label_column_name, positive_label, negative_label)),
            test=Dataset.from_dict(
                get_string_data(test_data, question, label_column_name, positive_label, negative_label)),
        )

        def preprocess_data(examples):
            model_inputs = self.tokenizer(examples['texts'], max_length=self.max_input_length, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples["labels"], max_length=self.max_target_length,
                                        truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(preprocess_data, batched=True)
        return tokenized_datasets, test_data

    def preprocess_data(self, examples):
        model_inputs = self.tokenizer(examples['texts'], max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["labels"], max_length=self.max_target_length,
                                    truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
