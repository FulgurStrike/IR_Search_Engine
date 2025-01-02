import os
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
import string
import spacy


nlp = spacy.load("en_core_web_sm")

stops = set(stopwords.words('english'))

vocab = {}
postings = {}
docsID = {}


def createDocsID():
    directory = "videogames/ps2.gamespy.com"
    count = 0
    for path, folders, files in os.walk(directory):
        for filename in sorted(files):
            docsID[count] = {}
            docsID[count]["name"] = (f"{filename}")
            count += 1


def detectBigrams(text):
    allBigrams = list(nltk.bigrams(text))
    frequentBigrams = []
    freqdist = nltk.FreqDist(allBigrams)
    for bigram in allBigrams:
        if freqdist[bigram] >= 10:
            frequentBigrams.append(bigram)

    return frequentBigrams



def Tokenizor(docID, text):
    soup = BeautifulSoup(text, 'html.parser')
    body = soup.find("body")
    text = body.get_text(separator = " | ", strip = True)
    noPunct = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(noPunct) 
    lowercaseTokens = [token.lower() for token in tokens]
    docsID[docID]["count"] = len(lowercaseTokens)
    nlpDoc = nlp(" ".join(lowercaseTokens))
    counter = Counter(lowercaseTokens)
    cleanedTokens = [(token.lemma_, counter[token.text]) for token in nlpDoc]
    
    bigrams = detectBigrams(lowercaseTokens)
    counter = Counter(bigrams)
    for bigram in bigrams:
        cleanedTokens.append((bigram, counter[bigram]))

    return set(cleanedTokens)


def addToDict(tokenArray):
    if len(vocab) == 0:
        count = 0
    else:
        count = list(vocab.values())[-1] + 1
    for token in tokenArray:
        if token[0] not in vocab:
            vocab[token[0]] = count
            count += 1


def createPostingsAndVocab():
    for document in docsID:
        try:
            with open(f"videogames/ps2.gamespy.com/{docsID[document]['name']}", "r") as file:
                tokens = Tokenizor(document, file)
                addToDict(tokens)
                print(tokens)

                for token in tokens:
                    wordID = vocab.get(token[0])
                    if wordID not in postings:
                        postings[wordID] = []
                    if document not in postings[wordID]:
                        postings[wordID].append((document, token[1]))
                file.close()
        except Exception as e:
            print(f"Error opening file {e}")


createDocsID()
createPostingsAndVocab()



with open("docsID.pkl", "wb") as file:
    pickle.dump(docsID, file)

with open("vocab.pkl", "wb") as file:
    pickle.dump(vocab, file)

with open("postings.pkl", "wb") as file:
    pickle.dump(postings, file)
