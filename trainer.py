import numpy as np
import json
from processor import Processor
from transformers import TrainingArguments, AutoTokenizer, DistilBertForSequenceClassification
from transformers import TrainerCallback, EarlyStoppingCallback, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer


class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


class Trainer(Processor):

    def get_trainer(self, tokenized_datasets):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        # train_dataset = load_data_set(tokenizer)

        arguments = Seq2SeqTrainingArguments(
            output_dir="sample_hf_trainer",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=20,
            # evaluation_strategy="epoch",  # run validation at the end of each epoch
            save_strategy='no',
            do_eval=False,
            evaluation_strategy="no",
            learning_rate=2e-5,
            # load_best_model_at_end=True,
            seed=224,
        )

        # def compute_metrics(eval_pred):
        #     """Called at the end of validation. Gives accuracy"""
        #     logits, labels = eval_pred
        #     predictions = np.argmax(logits, axis=-1)
        #     # calculates the accuracy
        #     return {"accuracy": np.mean(predictions == labels)}

        def compute_metrics(eval_pred):
            # for T5
            """Called at the end of validation. Gives accuracy"""
            predictions, labels = eval_pred
            # Decode the predictions
            predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            # Calculate the accuracy
            return {"accuracy": np.mean([pred == label for pred, label in zip(predictions, labels)])}

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=arguments,
            train_dataset=tokenized_datasets['train'],
            # eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
            # eval_dataset=eval_dataset,  # change to test when you do your final evaluation!
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        # trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
        trainer.add_callback(LoggingCallback("sample_hf_trainer/log.jsonl"))
        return trainer
