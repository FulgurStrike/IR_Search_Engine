import pickle
from nltk.tokenize import word_tokenize
import spacy
import math
import string
import numpy as np


nlp = spacy.load("en_core_web_sm")

with open("docsID.pkl", "rb") as file:
    docsID = pickle.load(file)

with open("vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

with open("postings.pkl", "rb") as file:
    postings = pickle.load(file)


print(postings[vocab["ps2"]])

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




def eucLength(matrix):
    return math.sqrt(sum([x**2 for x in matrix]))

def sim(query, doc):
    numerator = np.dot(query, doc)
    queryLen = eucLength(query)
    docLen = eucLength(doc)

   

    denominator = docLen * queryLen
    print(f"num: {numerator} | denom: {denominator}")
    similarity = numerator / denominator

    return similarity

while True:
    query = input("What would you like to query? (type quit to exit) ")
    if query == "quit":
        break

    noPunct = query.translate(str.maketrans("", "", string.punctuation))
    tokenisedQuery = word_tokenize(noPunct.lower())
    nlpDoc = nlp(" ".join(tokenisedQuery))
    lemmatizedQuery = [query.lemma_ for query in nlpDoc]
    
    try:

        rankedDocs = []
        if len(lemmatizedQuery) == 1:
            term = lemmatizedQuery[0]
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
            for tokenQuery in lemmatizedQuery:
                if tokenQuery in vocab:
                    vocabID = vocab[tokenQuery]
                    documents = postings[vocabID]
                    results[vocabID] = documents

            finalisedDocList = set()
            for term in lemmatizedQuery:
                vocabID = vocab[term]
                documents = results[vocabID]
                documentIDs = {doc[0] for doc in documents if doc[1] > 0}
                finalisedDocList = finalisedDocList.union(documentIDs)

            print(finalisedDocList)

            docVector = {}
            queryVector = generateQueryVector(lemmatizedQuery)

            for document in finalisedDocList:
                docVector[document] = []
                for term in lemmatizedQuery:
                    vocabID = vocab[term]
                    docVector[document].append(generateDocTFIDF(document, vocabID))

            
            for document in docVector:
                similarity = sim(queryVector, docVector[document])
                rankedDocs.append((document, similarity))


        rankedDocs = sorted(rankedDocs, key=lambda x: x[1], reverse=True)
        print("\n")
        for document in rankedDocs[:10]:
            print(f"{docsID[document[0]]['name']} | {document[1]}")
    
    except Exception as e:
        print(f"Word(s) not found try again! {e}")
