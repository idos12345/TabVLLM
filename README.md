# Tabular Data Classification with Large Language Models

## Overview
This project explores the optimal strategy for handling tabular data classification tasks using Large Language Models (LLMs). We aim to enhance and expand upon the research presented in the paper *"TabLLM: Few-shot Classification of Tabular Data with Large Language Models"*. Our approach involves experimenting with zero-shot and few-shot learning techniques and analyzing various label sets to determine the most effective ones for tabular data classification.

## Objectives
- Investigate the performance of zero-shot and few-shot learning for tabular data classification.
- Analyze different label sets to determine which are most effective for classification tasks.
- Develop a pipeline that can be applied to various Kaggle competitions.
- Ensure that our approach is competitive enough to achieve top results in tabular data classification tasks.

## Methodology
1. **Data Selection**: We will select diverse tabular datasets from Kaggle and other sources to ensure broad applicability.
2. **Model Experimentation**: Testing different LLMs for classification using zero-shot and few-shot learning.
3. **Label Optimization**: Evaluating various label sets to determine the most effective ones.
4. **Pipeline Development**: Creating a reusable and scalable pipeline that can be applied to various tasks.
5. **Performance Benchmarking**: Comparing our results with existing benchmarks and Kaggle competition winners.

## Expected Outcomes
- A deeper understanding of the effectiveness of LLMs in tabular data classification.
- A robust and flexible pipeline for handling tabular data tasks.
- Insights into the best label selection strategies for different datasets.
- Competitive results in Kaggle competitions and other benchmark datasets.

## How to Use

1. add your open AI key to conf.json
2. create a virtual environment 
3. run pip install -r requirements.txt
4. run python main.py (use --help to get argument description)
5. You can add more datasets to dataset_jsons and dataset_cves

You can also download colab_notebook.ipynb and run using Google Colab
