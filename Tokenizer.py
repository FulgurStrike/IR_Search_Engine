import os
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
import string
import spacy


nlp = spacy.load("en_core_web_lg")

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



def preProcessing(data, zoneWeighting, nerWeighting):
    noPunct = data.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(noPunct.lower()) 
    
    nlpDoc = nlp(" ".join(tokens))
    namedEntities = nlpDoc.ents
    lemmatizedTokens = [token.lemma_ for token in nlpDoc]
    cleanedTokens = Counter(lemmatizedTokens)

    for ent in nlpDoc.ents:
        nerWeight = nerWeighting.get(ent.label_, 1)
        cleanedTokens[ent.lemma_] *= nerWeight

    for token in cleanedTokens:
        cleanedTokens[token] *= zoneWeighting

    return cleanedTokens


def Tokenizor(docID, text):
    soup = BeautifulSoup(text, 'html.parser')
    
    body = soup.find("div", {"id": "content"}).get_text(separator = " | ", strip = True) 
    title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
    contentTitles = soup.find_all(class_="contenttitle")
    contentTitlesText = "".join(contenttitle.get_text(strip = True) for contenttitle in contentTitles)

    nerWeightings = {"ORG": 3, "PRODUCT": 2, "GPE": 2, "PERSON": 1.5, "DATE": 1.2} 
    
    bodyTokens = preProcessing(body, 1, nerWeightings)
    contentTitleTokens = preProcessing(contentTitlesText, 2, nerWeightings)
    titleTokens = preProcessing(title, 3, nerWeightings)

    combinedTokens = bodyTokens + contentTitleTokens + titleTokens

    docsID[docID]["count"] = sum(combinedTokens.values())
    
    return list(combinedTokens.items())


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
