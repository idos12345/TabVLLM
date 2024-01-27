from openai import OpenAI
import zipfile
import os
from utilities import *


def get_label_categories(columns):
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"We need to solve a classifcation problem, "
                                        f"We have a database with the following columns {', '.join(columns)}, where {columns[-1]} is the outcome column we want to predict."
                                        "Please generate carefully the best labels to use on the column we want to predict and a question, please give me 10 options for that."
                                        "IMPORTANT: for each line, Give only the question and the labels seperated by comma\n"
                                        "give it in a format like of: \n"
                                        "1. question, label1, label2"
             },
        ]

    )
    response_message = response.choices[0].message.content.split('\n')
    label_categories = []
    for line in response_message:
        if '.' in line:
            line = line[line.find('.') + 1:]
        label_categories.append([])
        for word in line.split(','):
            label_categories[-1].append(word.strip())
        assert len(label_categories[-1]) == 2
    return label_categories

class Preprocessor:

    def run(self, dataset, method):
      assert method in ["true-false", "yes-no", "llm-based"]

      if method == "true-false":
        dataset["positiveLabel"] = "true"
        dataset["negativeLabel"] = "false"
      elif method == "yes-no":
        dataset["positiveLabel"] = "yes"
        dataset["negativeLabel"] = "no"
      elif method == "llm-based":
        question, pos_label, neg_label = get_label_categories(dataset)
        dataset["question"] = question
        dataset["positiveLabel"] = pos_label
        dataset["negativeLabel"] = neg_label

    def processDataSet(self,datasetPath, label_column_name):
         data = pd.read_csv(datasetPath)
         if len(data) > 5000:
             data = data.head(5000)
         train_data = data.sample(frac=0.8, random_state=25)  # 80% for training
         test_data = data.drop(train_data.index)   # 20% for testing

         dataset = DatasetDict(
             train = Dataset.from_dict(get_string_data(train_data, label_column_name)),
             test = Dataset.from_dict(get_string_data(test_data, label_column_name)),
        )

    def preprocess_data(self,examples):
        model_inputs = tokenizer(examples['texts'], max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
          labels = tokenizer(examples["labels"], max_length=max_target_length,
                            truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
