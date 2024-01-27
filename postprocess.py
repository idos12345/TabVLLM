from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from utilities import *


def calculate_statistics(y_true, y_score):
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)

    return precision, recall, f1


class PostProcessor:

    def run_test_data(self, test_data, inputSuffix, label_column_name):
        test_data_set = get_string_data(test_data, label_column_name)
        correct = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i, text in enumerate(test_data_set['texts']):
            label = test_data_set['labels'][i]
            inputs = tokenizer([text + inputSuffix], max_length=max_input_length, truncation=True,
                               return_tensors="pt").to('cuda')
            output = model.generate(**inputs, num_beams=16, do_sample=True, min_length=0, max_length=64)
            # probs = softmax(output.float(), dim=-1)
            # y_score_temp = probs[:, 1].tolist()[0]
            # y_score.append(y_score_temp)
            label_string = label.strip()
            labelInt = 1 if label_string == positiveLabel else 0
            y_true.append(labelInt)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            output_string = decoded_output.strip()
            outputInt = 1 if output_string == positiveLabel else 0
            y_score.append(outputInt)
            if output_string == label_string:
                correct += 1
                if label_string == positiveLabel:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if output_string == positiveLabel:
                    false_positive += 1
                else:
                    false_negative += 1
                # print(f"{decoded_output} is not euqal {label}")
        length = len(test_data_set['texts'])
        print(f"predicted {correct} out of {length}. percentage {((correct / length) * 100)}")
