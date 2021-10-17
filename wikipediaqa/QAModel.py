from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BigBirdForQuestionAnswering
import numpy as np


class QAModel:
    def __init__(self, model=None, tokenizer=None, batch_size=4, device="cpu"):
        if not model:
            model = "twmkn9/distilbert-base-uncased-squad2"

        if not tokenizer:
            tokenizer = "twmkn9/distilbert-base-uncased-squad2"

        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.model = self.model.to(device)

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device

    def __call__(self, question, texts):  # predict
        # TODO separating by batches doesn't seem to solve the problem with RAM
        # try:
        preds = []
        for tokens in self.dataLoader(question, texts):
            tokens.to(self.device)

            logits = model(**tokens)

            starts, ends = logits[0], logits[1]

            for i, (start, end) in (enumerate(zip(starts, ends))):
                start, end = start.argmax(), end.argmax()
                answer_ids = tokens['input_ids'][i, start : end + 1]
                
                score = (start.max() + end.max()) / 2
                score = score.cpu().detach().numpy()

                answer = self.tokenizer.convert_ids_to_tokens(answer_ids)
                answer = ' '.join(answer)
                preds.append([answer, score])
            
        return preds

        # except Exception:
        #     print(traceback.format_exc())
        #     return [['Cuda:(', 0]]

    def tokenize(self, question, texts):
        questions = np.repeat(question, len(texts))
        self.inputs = self.tokenizer(questions.tolist(),
                                     texts,
                                     add_special_tokens=True,
                                     padding=True, truncation=True,
                                     return_tensors="pt")
        
        return self.inputs

    def dataLoader(self, question, texts):
        # TODO might be a better way to do it
        num_of_texts = len(texts)
        iteration = 0

        while True:
            texts_for_iteration = texts[
                iteration * self.batch_size : (iteration + 1) * self.batch_size]
            
            iteration += 1

            if texts_for_iteration:
                data = self.tokenize(question, texts_for_iteration)
                yield data

            else:
                break
