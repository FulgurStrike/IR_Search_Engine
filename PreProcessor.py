import os
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
import string
import spacy

class PreProcessor:
    
    nlp = spacy.load("en_core_web_lg")

    stops = set(stopwords.words('english'))

    vocab = {}
    postings = {}
    docsID = {}


    def createDocsID(self):
        directory = "videogames/ps2.gamespy.com"
        count = 0
        for path, folders, files in os.walk(directory):
            for filename in sorted(files):
                self.docsID[count] = {
                    "name": (f"{filename}"),
                    "count": self.totalWordsInDoc(f"{filename}")
                }
                count += 1


    def totalWordsInDoc(self, file):
        with open(f"videogames/ps2.gamespy.com/{file}") as file:
            soup = BeautifulSoup(file, "html.parser")
            body = soup.find("body").get_text(separator = " | ", strip = True)
            noPunct = noPunct = body.translate(str.maketrans("", "", string.punctuation))
            tokens = word_tokenize(noPunct.lower())
        file.close()

        return len(tokens)


    def preProcessing(self, data, zoneWeighting, nerWeighting):
        noPunct = data.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(noPunct.lower()) 
        nlpDoc = self.nlp(" ".join(tokens))
        namedEntities = nlpDoc.ents
        lemmatizedTokens = [token.lemma_ for token in nlpDoc]
        cleanedTokens = Counter(lemmatizedTokens)

        for ent in nlpDoc.ents:
            nerWeight = nerWeighting.get(ent.label_, 1)
            cleanedTokens[ent.lemma_] *= nerWeight

        for token in cleanedTokens:
            cleanedTokens[token] *= zoneWeighting

        return cleanedTokens


    def docPreProcessor(self, docID, text):
        soup = BeautifulSoup(text, 'html.parser')

        body = soup.find("div", {"id": "content"}).get_text(separator = " | ", strip = True) 
        title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
        contentTitles = soup.find_all(class_="contenttitle")
        contentTitlesText = "".join(contenttitle.get_text(strip = True) for contenttitle in contentTitles)

        nerWeightings = {"ORG": 3, "PRODUCT": 2, "GPE": 2, "PERSON": 1.5, "DATE": 1.2} 

        bodyTokens = self.preProcessing(body, 1, nerWeightings)
        contentTitleTokens = self.preProcessing(contentTitlesText, 5, nerWeightings)
        titleTokens = self.preProcessing(title, 3, nerWeightings)

        combinedTokens = bodyTokens + contentTitleTokens + titleTokens

        return list(combinedTokens.items())


    def addToDict(self, tokenArray):
        if len(self.vocab) == 0:
            count = 0
        else:
            count = list(self.vocab.values())[-1] + 1
        for token in tokenArray:
            if token[0] not in self.vocab:
                self.vocab[token[0]] = count
                count += 1


    def createPostingsAndVocab(self):
        for document in self.docsID:
            try:
                with open(f"videogames/ps2.gamespy.com/{self.docsID[document]['name']}", "r") as file:
                    tokens = self.docPreProcessor(document, file)

                    self.addToDict(tokens)

                    for token in tokens:
                        wordID = self.vocab.get(token[0])
                        if wordID not in self.postings:
                            self.postings[wordID] = []
                        if document not in self.postings[wordID]:
                            self.postings[wordID].append((document, token[1]))

                    file.close()
            except Exception as e:
                print(f"Error opening file {e}")

    def run(self):
        self.createDocsID()
        self.createPostingsAndVocab()

        with open("docsID.pkl", "wb") as file:
            pickle.dump(self.docsID, file)

        with open("vocab.pkl", "wb") as file:
            pickle.dump(self.vocab, file)

        with open("postings.pkl", "wb") as file:
            pickle.dump(self.postings, file)
