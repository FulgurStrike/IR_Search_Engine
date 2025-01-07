import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import math
import string
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from bs4 import BeautifulSoup

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')


nlp = spacy.load("en_core_web_sm")

with open("docsID.pkl", "rb") as file:
    docsID = pickle.load(file)

with open("vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

with open("postings.pkl", "rb") as file:
    postings = pickle.load(file)


def calculateTFIDF(termCount, totalWords, df):
    tf = (termCount) / (totalWords)
    wf = math.log(tf + 1)

    n = list(docsID.keys())[-1] + 1
    df = df
    idf = math.log(n / df) if df > 0 else 0

    wtf_idf = wf * idf

    return wtf_idf

def generateQueryVector(lemmatizedQuery):
    queryVector = []
    for term in lemmatizedQuery:
        totalWords = len(lemmatizedQuery)
        termCount = 1
        df = len(postings[vocab[term]])
        tfIDF = calculateTFIDF(termCount, totalWords, df)
        queryVector.append(tfIDF)

    return queryVector

def generateDocTFIDF(document, vocabID):
    documentVector = []
    posting = postings[vocabID]
    totalWordsInDoc = docsID[document]["count"]
    termCountDoc = next((x[1]  for x in posting if x[0] == document), 0)
    df = len(posting)
    tfIDF = calculateTFIDF(termCountDoc, totalWordsInDoc, df)

    return tfIDF

def queryExpansion(tokenisedQuery):
    tags = pos_tag(tokenisedQuery)

    nouns = [tag[0] for tag in tags if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

    sysnetObjects = [wordnet.synsets(noun) for noun in nouns]

    lemmas = [lemma for lemmaList in sysnetObjects for obj in lemmaList for lemma in obj.lemmas()]

    expandedQuery = set(tokenisedQuery + [lemma.name() for lemma in lemmas])

    filteredQuery = list(filter(lambda term: term in vocab.keys(), expandedQuery))

    return filteredQuery


def eucLength(matrix):
    return math.sqrt(sum([x**2 for x in matrix]))

def sim(query, doc):
    numerator = np.dot(query, doc)
    queryLen = eucLength(query)
    docLen = eucLength(doc)

    denominator = docLen * queryLen
    similarity = numerator / denominator

    return similarity

def display(file):
    file = open(f"videogames/ps2.gamespy.com/{file}")
    soup = BeautifulSoup(file, 'html.parser')
    body = soup.find("div", {"id": "content"}).get_text(separator = " | ", strip = True)

    sentences = sent_tokenize(body)
    content = " ".join(sentences[:3]).replace("|", "").strip()

    file.close()

    return content


while True:
    query = input("What would you like to query? (type quit to exit) ")
    if query == "quit":
        break

    noPunct = query.translate(str.maketrans("", "", string.punctuation))
    tokenisedQuery = word_tokenize(noPunct.lower())
    expandedQuery = queryExpansion(tokenisedQuery)
    print(expandedQuery)

    try:

        rankedDocs = []
        if len(expandedQuery) == 1:
            term = expandedQuery[0]
            vocabID = vocab[term]
            documents = postings[vocabID]
            for document in documents:
                docId = document[0]
                totalWords = docsID[docId]["count"]
                termCount = document[1]
                df = len(documents)
                tfIDF = calculateTFIDF(termCount, totalWords, df)
                rankedDocs.append((docId, tfIDF))
        else: 
            results = {}
            for tokenQuery in expandedQuery:
                if tokenQuery in vocab:
                    vocabID = vocab[tokenQuery]
                    documents = postings[vocabID]
                    results[vocabID] = documents

            finalisedDocList = set()
            for term in expandedQuery:
                vocabID = vocab[term]
                documents = results[vocabID]
                documentIDs = {doc[0] for doc in documents if doc[1] > 0}
                finalisedDocList = finalisedDocList.union(documentIDs)

            docVector = {}
            queryVector = generateQueryVector(expandedQuery)

            for document in finalisedDocList:
                docVector[document] = []
                for term in expandedQuery:
                    vocabID = vocab[term]
                    docVector[document].append(generateDocTFIDF(document, vocabID))
            
            for document in docVector:
                similarity = sim(queryVector, docVector[document])
                rankedDocs.append((document, similarity))

        rankedDocs = sorted(rankedDocs, key=lambda x: x[1], reverse=True)
        print("\n")
        for document in rankedDocs[:10]:
            print(f"{docsID[document[0]]['name']} | {document[1]}\n{display(docsID[document[0]]['name'])}\n")
    
    except Exception as e:
        print(f"Word(s) not found try again! {e}")
