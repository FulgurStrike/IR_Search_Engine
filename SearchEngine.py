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

def calculateTFIDF(termCount, totalWords, vocabID):
    tf = (termCount) / (totalWords)
    wf = math.log(tf + 1)

    n = list(docsID.keys())[-1] + 1
    df = len(postings[vocabID])
    idf = math.log((n + 1)  / (df + 1)) + 1

    wtf_idf = wf * idf

    return wtf_idf


def eucLength(matrix):
    return math.sqrt(sum([x**2 for x in matrix]))

def sim(query, doc):
    numerator = np.dot(query, doc)
    queryLen = eucLength(query)
    docLen = eucLength(doc)

    denominator = docLen * queryLen

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
            documentIDs = {doc[0] for doc in documents}
            finalisedDocList = finalisedDocList.union(documentIDs)
            
        docVector = {}
        queryVector = {}
        rankedDocs = []
        for document in finalisedDocList:
            docVector[document] = []
            for term in lemmatizedQuery:
                queryVector[term] = []
               
                vocabID = vocab[term]
                posting = postings[vocabID]

                totalWordsInDoc = docsID[document]["count"]
                termCountDoc = next((x[1] for x in posting if x[0] == document), 0)
                
                totalWordsInQuery = len(query)
                termCountInQuery = 1

                tfIdfQuery = calculateTFIDF(termCountInQuery, totalWordsInQuery, vocabID)
                tfIdfDoc = calculateTFIDF(termCountDoc, totalWordsInDoc, vocabID)


                docVector[document].append(tfIdfDoc)
                queryVector[term].append(tfIdfQuery)

        for query in queryVector:
            queryVector[query] = list(np.pad(
                queryVector[query], 
                (0, len(docVector[document]) - len(queryVector[query])), 
                mode = "constant"))
            for document in docVector:
                similarity = sim(queryVector[query], docVector[document])
                rankedDocs.append((document, similarity))


        
       
        print(queryVector)
        print(docVector)
        rankedDocs = sorted(rankedDocs, key=lambda x: x[1], reverse=True)

        for document in rankedDocs[:10]:
            print(f"\n{docsID[document[0]]['name']} | {document[1]}")
    
    except Exception as e:
        print(f"Word(s) not found try again! {e}")
