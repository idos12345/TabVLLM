import preprocess, postprocess
from utilities import *
from trainer import *


class Pipeline:
    def __init__(self):
        self.model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.max_input_length = 512
        self.max_target_length = 64

    def run(self, dataset, method):
        y_true = []  # true labels
        y_score = []  # predicted scores
        csv_file_path = get_data_set_from_kaggle(dataset)

        with preprocess.Preprocessor() as pre_processor:
            pre_processor.run(dataset, method)
            positiveLabel = dataset["positiveLabel"] if dataset.get("positiveLabel") else "Yes"
            negativeLabel = dataset["negativeLabel"] if dataset.get("negativeLabel") else "No"
            test_data = None
            inputSuffix, maxLength = set_dataset_params(dataset, positiveLabel, negativeLabel)
            tokenized_datasets, test_data = pre_processor.processDataSet(csv_file_path, dataset['labelColumnName'])

        trainer = get_trainer(tokenized_datasets)
        trainer.train()
        trainer.save_model()

        with postprocess.PostProcessor() as post_processor:
            post_processor.run_test_data(test_data, inputSuffix, dataset['labelColumnName'])
            post_processor.calculate_statistics()
