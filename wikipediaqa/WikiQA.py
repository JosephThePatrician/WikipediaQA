import torch
import warnings


class WikiQA:
    def __init__(
            self, 
            model_name=None, 
            sentenceModel_name=None,
            lang = "en", 
            batch_size=4, 
            device=None):

        """
        model_name: path to model or Hugging Face model name
        model should be for question answering
        You can choose a model from https://huggingface.co/models
        default: "twmkn9/distilbert-base-uncased-squad2"
        
        sentenceModel_name: path to model or Hugging Face model name
        model should be for sentence similarity
        You can choose a model from https://huggingface.co/models
        default: "sentence-transformers/paraphrase-distilroberta-base-v2"

        lang: language of wikipedia
        list of all languages https://meta.wikimedia.org/wiki/List_of_Wikipedias
        default "en"
        """

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if (not model_name) and (lang == "en"):
            warnings.warn(
                "You have changed the language but did not changed the model\n"
                "You should choose the model for the right language from https://huggingface.co/models", 
                RuntimeWarning)

        self.model = QAModel(model_name, model_name, batch_size, device)
        
        self.sentenceModel = SentenceModel(sentenceModel_name, device)

        self.parser = WikiParser()
        self.textProcessor = TextProcessor()

    def __call__(self, question):
        # try:
        answers = []
        search_requests = self.textProcessor(question)

        for search_request in search_requests:
            answers.extend(self.ask(question, search_request))
        
        if answers:
            # # debug
            # sort = sorted(answers, key=lambda x: x[1])
            # sort = '\n'.join([str(i[1]) + ' ' + str(i[0]) for i in sort])
            # print('\n', sort, '\n')

            answer = self.best_answer(answers)
            return self.textProcessor.postprocess_answer(answer)
        
        # print('you had only one job and failed')

        return "Can't find answer :("

        # except Exception:
        #     print(traceback.format_exc())
        #     return 'Something went wrong :('

    def ask(self, question, search_request):
        # print('we search wiki for:', search_request)
        search = self.parser.search(search_request)
        
        # # debug
        # print("\n search")
        # print('\n'.join([i.title for i in search]))
        # print('\n')

        if len(search) == 0:  # if nothing have been found
            print('wiki finds nothing')
            return [['wiki finds nothing :(', 0]]

        # summarys = [i.summary() for i in search if 'may refer to' not in i]
        summarys = []
        for i in search:
            summary = i.summary()
            # remove pages like this https://en.wikipedia.org/wiki/Python
            if "refer to" not in summary:
                summarys.append(summary)


        # find the best page via sentenceModel
        best = self.sentenceModel.compare(question, summarys)
        bestPage = search[best]
        
        # debug
        # print(f"bestPage {bestPage.title} {bestPage.url()}")

        answers = self.getAnswers(question, bestPage)

        return answers

    def askFast(self, question, page, url):
        """
        Process only through summary and infoBox
        """

        info = self.parser.getInfo(url)  # get infoBox from page
        summary = page.summary()  # get summary from page

        texts = [info] + [summary]

        texts = [i for i in texts if len(i) != 0]  # remove empty texts

        # tokens = self.tokenize(self.question, texts)

        answers = self.model(question, texts)

        return answers

    def askSlow(self, question, url):
        """
        Process all the text on the page
        """
        
        # info = self.parser.getInfo(url)  # get infoBox from page
        texts = self.parser.getText(url)  # get all the text from page

        texts = self.findEnts(question, texts)

        # texts = [info] + texts

        texts = [i for i in texts if len(i) > 5]  # remove empty texts

        answers = self.model(question, texts)

        return answers

    def getAnswers(self, question, page):
        url = page.url()

        answers = self.askFast(question, page, url)
            
        good_answers = self.find_good_answers(answers)

        if len(good_answers) > 0:  # if there are good answers
            return good_answers

        # otherwise ask to look better
        answers = self.askSlow(question, url)

        good_answers = self.find_good_answers(answers)
        
        if len(good_answers) != 0:  # if there are good answers
            return good_answers

        # otherwise
        # print("can't find answer for that")
        return []

    def best_answer(self, answers):
        answers = sorted(answers, key=lambda x: x[1])
        return answers[-1][0]

    def findEnts(self, question, texts):
        """
        Finds paragraphs with entities from the question
        """
        # print(f'Entities: {ents}')
        # labels = [ent.label_ for ent in ents]
        ents = self.textProcessor.getAllNE(question)
        # print("found ents:", ents)
        pars = []

        for ent in ents:
            for par in texts:
                if str(ent).lower() in par.lower():
                    # pars.append(ent)
                    pars.append(par)

        return pars

    def find_good_answers(self, answers):
        """
        removes meaningless answers
        """
        return [[answer, score] for answer, score in answers if
                        ('[CLS]' not in answer) and
                        ('[SEP]' not in answer) and
                        ('<pad>' not in answer) and
                        ('<unk>' not in answer) and
                        (len(answer) < 200) and
                        (len(answer) > 2) and
                        (answer != '') and
                        (score > 1)]