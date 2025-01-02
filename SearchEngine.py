import pickle
from nltk.tokenize import word_tokenize
import spacy
import math

nlp = spacy.load("en_core_web_sm")

with open("docsID.pkl", "rb") as file:
    docsID = pickle.load(file)

with open("vocab.pkl", "rb") as file:
    vocab = pickle.load(file)

with open("postings.pkl", "rb") as file:
    postings = pickle.load(file)

while True:
    query = input("What would you like to query? (type quit to exit) ")
    if query == "quit":
        break
    tokenisedQuery = word_tokenize(query)
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
            

        rankedDocs = []
        for document in finalisedDocList:
            tf_idfScores = []
            for term in lemmatizedQuery:
               
                vocabID = vocab[term]
                posting = postings[vocabID]

                totalWords = docsID[document]["count"]
                termCount = next((x[1] for x in posting if x[0] == document), 0)
                tf = (termCount) / (totalWords)
                wf = math.log(tf + 1)

                n = list(docsID.keys())[-1] + 1
                df = len(postings[vocabID])
                idf = math.log((n + 1)  / (df + 1)) + 1

                wtf_idf = wf * idf
                tf_idfScores.append(wtf_idf)

                print(f"TF: {tf}, WF: {wf}, IDF: {idf}, WF-IDF: {wtf_idf}")
            tfIDFForCurrentDoc = sum(tf_idfScores)
            rankedDocs.append((document, tfIDFForCurrentDoc))

        rankedDocs = sorted(rankedDocs, key=lambda x: x[1], reverse=True)

        for document in rankedDocs[:10]:
            print(f"{docsID[document[0]]['name']} | {document[1]}")
    
    except Exception as e:
        print(f"Word(s) not found try again! {e}")
