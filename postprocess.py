from processor import Processor
from utilities import *


class PostProcessor(Processor):

    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, model)
        self.y_true = []
        self.y_score = []

    def run_test_data(self, test_data, question, label_column_name, positive_label, negative_label):
        test_data_set = get_string_data(test_data, label_column_name, question, positive_label, negative_label)
        correct = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i, text in enumerate(test_data_set['texts']):
            label = test_data_set['labels'][i]
            inputs = self.tokenizer([text], max_length=self.max_input_length, truncation=True, return_tensors="pt").to(
                'cuda')
            output = self.model.generate(**inputs, num_beams=16, do_sample=True, min_length=0, max_length=64)
            label_string = label.strip()
            labelInt = 1 if label_string == positive_label else 0
            self.y_true.append(labelInt)
            decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            output_string = decoded_output.strip()
            outputInt = 1 if output_string == positive_label else 0
            self.y_score.append(outputInt)
            if output_string == label_string:
                correct += 1
                if label_string == positive_label:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if output_string == positive_label:
                    false_positive += 1
                else:
                    false_negative += 1
                # print(f"{decoded_output} is not euqal {label}")
        length = len(test_data_set['texts'])
        print(f"predicted {correct} out of {length}. percentage {((correct / length) * 100)}")

    def calculate_statistics(self):
        precision = precision_score(self.y_true, self.y_score)
        recall = recall_score(self.y_true, self.y_score)
        f1 = f1_score(self.y_true, self.y_score)

        return precision, recall, f1
