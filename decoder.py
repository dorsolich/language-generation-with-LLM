from config import get_logger, device
_logger = get_logger(logger_name=__name__)
_logger.info(f"""Running in device: {device}""")



class Decoder:
    def __init__(self, device) -> None:
        self.device = device
        self.contexts = []
        self.questions = []
        self.generated_questions = []

    def tokenize(self,
                example,
                tokenizer, 
                max_length_source=512, 
                truncation=True, 
                padding="max_length",
                return_tensors = "pt"
                ):
        
        self.tokenizer=tokenizer
        self.question, self.context = example["question"], example["context"].replace('\n', '') # do this in preprocessing
        encodings = self.tokenizer(self.context, truncation=truncation, max_length=max_length_source, padding=padding, return_tensors=return_tensors)
        self.input_ids = encodings['input_ids'].to(self.device)
        self.input_attention_mask = encodings['attention_mask'].to(self.device)

        return self

    def decode(self, 
                model, 
                num_beams,  
                question_max_length=32, 
                repetition_penalty=2.5,
                length_penalty=1,
                early_stopping=True,
                use_cache=True,
                num_return_sequences=1,
                do_sample=False,
                ):

        generated_ids = model.generate(
            input_ids=self.input_ids,
            attention_mask=self.input_attention_mask,
            num_beams=num_beams, 
            do_sample=do_sample,
            max_length=question_max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences
        )
        for generated_id in generated_ids:
            generated_questions = self.tokenizer.decode(generated_id,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True
                                                        )

            self.generated_questions.append(generated_questions)
            self.contexts.append(self.context)
            self.questions.append(self.question)

        return self