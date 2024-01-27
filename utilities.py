import os

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import zipfile

def calculate_statistics(y_true, y_score):
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


def get_data_set_from_kaggle(info):
    owner = info["owner"]
    dataset = info["dataset"]
    csvFileName = info["csvFileName"]

    datasetFullName = f"{owner}/{dataset}"
    mainFolder = "datasets"

    if owner == "competitions":
        os.system(f"kaggle competitions download -c {dataset}")

        mainFolder = "."
    else:
        os.system(f"kaggle datasets download -d {datasetFullName}")

    datasetFolder = f"/content/kaggle/{mainFolder}/{owner}/{dataset}"
    csv_file_path = f"{datasetFolder}/{csvFileName}"

    with zipfile.ZipFile(f"{datasetFolder}/{dataset}.zip", "r") as zip_ref:
        zip_ref.extract(f"{csvFileName}", datasetFolder)  # Extract to a specific directory
    return csv_file_path


def set_dataset_params(question, positiveLabel, negativeLabel):
    question = question
    inputSuffix = f"\nPlease answer the following question.\n {question} {positiveLabel} or {negativeLabel}?"
    maxLength = max(len(positiveLabel), len(negativeLabel))
    return inputSuffix, maxLength


def get_string_data(data_frame, label_column_name, positiveLabel, negativeLabel):
    texts = []
    labels = []
    for index, row in data_frame.iterrows():
        # Construct the formatted string for the current row
        row_string = ', '.join([f'{column}: {value}' for column, value in row.items() if column != label_column_name])

        texts.append(row_string)
        labelInt = int(row[label_column_name])
        labelText = positiveLabel if labelInt == 1 else negativeLabel
        labels.append(labelText)

    data_set = {
        'texts': texts,
        'labels': labels,
    }

    return data_set
