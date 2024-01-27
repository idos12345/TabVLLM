import click
from pipeline import Pipeline

@click.command()
@click.option('--dataset_json', help='location of the dataset json', required=True)
@click.option('--method', help='yes-no/true-false/llm-based', required=True)
def main(dataset_json, method):
    pipeline = Pipeline()
    pipeline.run(dataset_json, method)

if __name__ == '__main__':
    main()