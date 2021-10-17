import numpy as np
import spacy
import re
from warnings import warn


class TextProcessor:
    def __init__(self, spacy_model : str = "en_core_web_sm"):
        # try:
        self.nlp = spacy.load(spacy_model)
        # except Exception:
        #     raise Exception(f"{spacy_model} was not found, you should install it via python -m spacy {spacy_model}")

    def __call__(self, question : str):
        """
        Extract everything helpful for search queries
        """
        question = self.removeExtraSpaces(question)

        doc = self.nlp(question)

        important_stuff = []

        inQuot = self.getInQuotation(question)
        important_stuff.extend(inQuot)

        ne = self.getNE(doc)
        important_stuff.extend([sent for sent in ne 
                                    if sent not in important_stuff])
        
        title = self.getTitle(question)
        important_stuff.extend([sent for sent in title 
                                    if sent not in important_stuff])

        nnp = self.getNNP(doc)

        # check if any NNPs have already been taken
        nnp = self.remove_repeating(nnp + ne + inQuot + title)
        important_stuff.extend([sent for sent in nnp
                                    if sent not in important_stuff])
        
        # if nothing is found, return question
        if len(important_stuff) == 0:
            return [question]
        
        return important_stuff

    def removeExtraSpaces(self, text):
        # remove multible spaces
        text = re.sub(' +', ' ', text)
        # remove spaces at the end and start of string
        text = re.sub(r'\A +| +$', '', text)
        return text

    def postprocess(self, text):
        # the is useless in search
        text = text.replace('the', '')
        text = self.removeExtraSpaces(text)

        return text

    def postprocess_answer(self, answer):
        """postrocess answer of model"""
        if '▁' in answer: # some models have this approach
            # replace "▁wo rd" with "word"
            answer = answer.replace(' ', '').replace('▁', ' ')

        if '##' in answer: # others models this
            # replace "wo ##rd" with "word"
            answer = answer.replace(' ##', '')

        # replace "word . Word" with "word. Word", same with comma
        answer = answer.replace(' .', '.').replace(' ,', ',')

        # replace "word ' s" with "word's"
        answer = answer.replace(' \' ', '\'')

        # replace floats "2. 3" and "2, 3" with "2.3" and "2,3"
        answer = re.sub(r'\d,\s\d|\d\.\s\d',
                        lambda x: x.group(0).replace(' ', ''), 
                        answer)
        
        answer = self.removeExtraSpaces(answer)

        return answer

    def remove_punctuation(self, text):
        text = re.sub(r"[.,!?;:()[]{}\"'«»]", '', text)
        return text

    def getTitle(self, text):
        """
        Get all words that start with upper letter
        example:
        "Hello World, it's me Mario" -> ["Hello World", "Mario"]
        """
        text = self.remove_punctuation(text)
        split = text[:-1].split()
        capital = [i if i[0].isupper() else '  ' for i in split[1:]]
        capital = ' '.join(capital)

        return [i for i in re.split('  +', capital) if i]

    def getNE(self, doc):
        """return every named entity, that can be used in search"""
        ne = [self.postprocess(ent.text) for ent in doc.ents 
                if ent.label_ not in ('CARDINAL', 'DATE', 'ORDINAL', 'NORP')]

        return ne

    def getAllNE(self, doc):
        """return every named entity"""
        if type(doc) == str:
            doc = self.nlp(doc)

        return doc.ents

    def getInQuotation(self, text):
        """
        return sentence in quotations if there are any
        """

        # get rid of 's, 've, etc.
        sent = re.sub(r"[a-zA-Z][\'][a-zA-Z]", '[quot]', text)

        # check if there are any quots
        if ('\'' not in sent) and ('\"' not in sent) and ('«' not in sent):
            return []

        # there might be words like "James' " that have quots in the end
        if ('\"' in sent or '«' in sent) and "\'" in sent:
            sent = sent.replace("\'", '[quot]')

        if sent.count("\'") % 2 == 1:
            warn(f"could not figure out the quotations in {text}")
            return []

        # find sentences in quotations
        sent = re.findall(r"(?<=').*(?=')|"
                          r'(?<=").*(?=")|'
                          r"(?<=«).*(?=»)", sent)

        # replace all quotes with "
        # wikipedia will search exact match for sentences in quotes
        sent = [f'"{i}"' for i in sent]

        sent = sent.replace('[quot]', "\'")  # put back '

        return [self.postprocess(i) for i in sent]

    def getNNP(self, doc):
        """return every proper noun or noun and everything associated with this words"""
        important = []
        for id, token in enumerate(doc):
            if (token.pos_ in ['PROPN', 'NOUN']):

                associated = self.find_associated(token, [])

                # do not return simple words like "name"
                if (len(associated) > 0 or
                        (len(associated) == 0 and token.text.istitle())):

                    important.append(associated + [token.i])

        return [' '.join([
                          self.postprocess(doc[i].text) for i in sorted(sent)
                            if 'name' not in doc[i].text and 
                               'many' not in doc[i].text and
                               'much' not in doc[i].text
                            ]) for sent in important
                ]


    def remove_repeating(self, important):
        """
        remove repeating words and parts of words
        example:
        ["Hello", "Hello World", "Hello World"] -> ["Hello World"]
        """
        # get amount of words
        len_important = [i.count(' ') for i in important]

        # sort by the amount of words in ascending order
        sort = np.argsort(len_important)
        important = np.array(important, dtype=object)[sort]


        l = len(important)

        # list of positions of words to remove
        leave = list(range(l))
        for i in range(l):
            for u in range(i+1, l):
                if important[i] in important[u]:
                    leave[i] = -1
                    break

        # which words to leave
        leave = [i for i in leave if i != -1]

        return important[leave].tolist()

    def find_associated(self, token, output):  # you need to pass output=[]
        """return everything connected with word"""
        for child in token.children:
            if child.dep_ in ['compound', 'nmod', 'nummod', 'amod']:
                output.append(child.i)
                self.find_associated(child, output)

        return output
