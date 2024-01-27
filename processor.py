class Processor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.max_input_length = 512
        self.max_target_length = 64