#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:42:48 2019

@author: Roman Salzwedel
"""

# 1. *Indexing*

### Libaries and Packages

# Using nltk lemmatizer requires to download the WordNetDictionary first!
import nltk
nltk.download('wordnet')

import re
import pandas as pd
import numpy as np
import math
import copy
from time import time
from functools import reduce 
from random import randint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer             

### 1.1 preprocess()

# List of regex to be tested for tokenization:
# Naive: "[a-z\-]+"
# More advanced: "(?u)\\b\\w\\w+\\b"
def preprocess(d, regex="(?u)\\b\\w\\w+\\b", stopwordList="data/raw/stopwords.large",
               lemmatizeTokens=False, stemmPorter=True, stemmSnowball=False):
    
    """Input: Single document d, regular expression that is used for tokenization, path to stopword list.
    Splits all terms in d according to given tokenizer, converts terms to lower case,
    removes stopwords. Returns a list of preprocessed terms"""

    terms = list()

    stopwords = open(stopwordList).read()
    stopwords = stopwords.split(sep="\n")
    tokenizer = re.compile(regex, re.IGNORECASE)  # letters a-z, -, and ''
    if lemmatizeTokens:
        lemmatizer = WordNetLemmatizer()
    if stemmSnowball:
        stemmer = SnowballStemmer(language='english')
    if stemmPorter:
        stemmer = PorterStemmer()

    for term in tokenizer.findall(d):

        # lower case
        term = term.lower()

        # remove stopwords
        if term not in stopwords:

            # if true lemmatize tokens
            if lemmatizeTokens:
                term = lemmatizer.lemmatize(term)

            # if true stemm tokens
            if stemmPorter or stemmSnowball:

                term = stemmer.stem(term)

            terms.append(term)

    return terms

### 1.2 compute_Tf()

def compute_Tf(d, rawScores=False, binary=False):
    
    """Input: Single document d. Calls preprocess(). For each preprocessed term t in document d it computes
    normalized (or raw) term-frequency of t in d. Returns dictionary"""

    Tfs = dict()

    # preprocess terms t calling preprocess():
    terms = preprocess(d)

    for term in terms:

        if term in Tfs and not binary:
            Tfs[term] += 1
        else:
            Tfs[term] = 1

    if rawScores:
        return Tfs

    # Find maximum Tf value:
    maximum = max(Tfs, key=Tfs.get)
    maxTf = 1 + math.log10(Tfs[maximum])

    # Normalize Tf scores using logarithm and divide by maximum Tf score
    for term in Tfs:

        if Tfs[term] > 0:

            Tfs[term] = (1 + math.log10(Tfs[term])) / (maxTf)

    return Tfs

### 1.3 compute_Idf()

def compute_Idf(D, tfDict=None, verbose=True):
    
    """Input: Document collection D. If tf scores are not provided 
    compute tf first. Then compute document frequency df and inverse document
    frequency idf for each term t in document collection D. Return dictionary with idf values"""

    t0 = time()

    Idfs = dict()      # dict that holds inverse-document frequency (idf) scores
    Dfs = dict()       # dict that holds document-frequency (df) scores
    Tfs = dict()       # dict that holds (normalized) term-frequency (tf) scores

    # If tf scores are not provided, call computeTf()
    if tfDict is None:
        for doc in D.keys():
            Tfs[doc] = compute_Tf(D[doc])
    else:
        Tfs = copy.deepcopy(tfDict)

    # Compute df for each term in document collection D
    for doc in Tfs.keys():
        for term in Tfs[doc]:

            if term in Dfs and Tfs[doc][term] > 0:
                Dfs[term] += 1
            else:
                Dfs[term] = 1

    # Compute number of document in the document collection!!!!
    N = len(D.keys())

    # Compute idf for each term
    for term in Dfs.keys():
        Idfs[term] = math.log10(N / Dfs[term])

    if verbose:
        print("Idf computation done in {:.4f}s.".format(time() - t0))

    return Idfs

### 1.4a compute_TfIdf()

def compute_TfIdf(D, idfDict=None, verbose=True):
    
    """Input: Document collection D. Optional: idf scores. Compute tf-idf values for each term 
    t across all documnets d in document collection D. Return dictionary with tf-idf scores."""

    t0 = time()

    TfIdfs = dict()
    Idfs = dict()
    Tfs = dict()

    # If idf scores are not provided call compute_Idf()
    if idfDict is None:
        Idfs = compute_Idf(D)
    else:
        Idfs = idfDict

    # Compute term-frequencies for all documents d in D
    for doc in D.keys():
        Tfs[doc] = compute_Tf(D[doc])

    # compute Tf-Idf scores for each t across all documents in D
    TfIdfs = copy.deepcopy(Tfs)

    for doc in D.keys():
        for term in Idfs:
            # check if term is part of document d
            if term in Tfs[doc]:
                # if tf > 0
                if Tfs[doc][term] > 0:
                    # tfidf = normalized tf * idf
                    TfIdfs[doc][term] = Tfs[doc][term] * Idfs[term]

                else:
                    TfIdfs[doc][term] = 0

            else:
                continue

    if verbose:
        print("Tf-idf computation done in {:.4f}s.".format(time() - t0))

    return TfIdfs

### 1.4b compute_TfIdfQuery()

def compute_TfIdfQuery(q, idfDict):
    
    """Input: Query q, and Idf dictionary. For each term t in query q compute tf-idf scores.
    Return dictionary of tf-idf scores."""

    Tfidf_query = dict()
    Idfs = idfDict

    # Compute term frequency for each term t in query q
    Tf_query = compute_Tf(q)

    # Transform tf scores into tf-idf scores using global idf for each term t
    for term in Tf_query.keys():

        if term in Idfs.keys():

            Tfidf_query[term] = Tf_query[term] * Idfs[term]

    return Tfidf_query

### 1.5 construct_inverted_index()

def construct_invertedIndex(D, idfDict=None, tfidfDict=None, verbose=True):
    
    """Input: Document collection D. For each document d in D compute Tf-Idf across all terms
    t in document d. For each term t add tuple(docID, tf-idf) to posting list in inverted index."""

    t0 = time()

    invertedIndex = dict()
    TfIdfs = dict()

    # compute idf scores
    if idfDict is None:
        Idfs = compute_Idf(D)
    else:
        Idfs = idfDict

    # compute tfidf scores
    if tfidfDict is None:
        TfIdfs = compute_TfIdf(D, Idfs)
    else:
        TfIdfs = tfidfDict

    # Add tuple(docID, tf-idf) to postinglist for each term in vocabulary
    for doc in D.keys():

        for term in TfIdfs[doc]:

            e = (doc, TfIdfs[doc][term])

            if term in invertedIndex.keys():
                invertedIndex[term].append(e)

            else:
                invertedIndex[term] = [e, ]

    if verbose:
        print("InvertedIndex construction done in {:.4f}s.".format(
            time() - t0))

    return invertedIndex

### 1.6 doc_length()

def construct_docLengthDict(D, tfidfDict=None, verbose=True):
    
    """Input: Document collection D. For each document d in D vector norm length of d.
    Return dictionary {docID : docLength}"""

    t0 = time()

    docLength = dict()

    if tfidfDict is None:
        tfidfDict = compute_TfIdf(D)

    for doc in tfidfDict.keys():

        docLength[doc] = 0

        for term in tfidfDict[doc]:

            docLength[doc] += math.pow(tfidfDict[doc][term], 2)

        docLength[doc] = math.sqrt(docLength[doc])

    if verbose:
        print("DocLength index construction done in {:.4f}s.".format(
            time() - t0))

    return docLength

### 1.7 create_tdm()

def create_tdm(D, tfidfDict=None, verbose=True):
    
    """Input: Document collection D. Call compute_TfIdf() and calculate tf-idf values for each term in 
    document collection. Return term-document matrix (tdm) as pd.DataFrame"""

    t0 = time()

    TfIdfs = dict()

    # call compute_TfIdf()
    if tfidfDict is None:
        TfIdfs = compute_TfIdf(D)
    else:
        TfIdfs = copy.deepcopy(tfidfDict)

    # convert into pandas data frame
    TfIdfs_df = pd.DataFrame(TfIdfs).fillna(0)

    if verbose:
        print("TDM construction done in {:.4f}s.".format(time() - t0))

    return TfIdfs_df

### 1.8 construct_tiered_index()

def construct_tiered_index(D, inv_index_dict=None, t=0.8, verbose=True):
    
    """Add some DocString here"""
    
    t0 = time()
    tieredIndex = dict()

    # compute inverted index
    if inv_index_dict is None:
        invertedIndex = construct_invertedIndex(D)
    else:
        invertedIndex = inv_index_dict

    for term in invertedIndex.keys():
        inv_index_list = invertedIndex[term]
        tier1, tier2 = [], []

        # Create 2 level tiered index
        for item in inv_index_list:

            if item[1] >= t:  # Set this level to decide on the split based on the TF- IDF score
                tier1.append(item)
            else:
                tier2.append(item)

            tier_list = [tier1, tier2]

        tieredIndex[term] = tier_list
        
    if verbose:
        print("TieredIndex construction done in {:.4f}s.".format(time() - t0))

    return tieredIndex



# 2. *QUERYING*

### 2.1 vanilla_cosine()

def vec_cosine(v1, v2):
    
    """Input: Two vectors. Calculate cosine score between both input vectors using numpy."""

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def vanilla_cosine(q, TDM, idfDict):
    
    """Input: Query q, Term-document matrix with tf-idf scores, Idf dictionary. Compute tf-idf scores
    for each term t in query q. Then append vector and add query q to tdm. Calculate cosine simalrity between
    each document d in D (each column in tdm) and query q. Store results and return dictionary."""

    Tfidf_query = dict()    
    Vector_q = dict()       
    scores = dict() 

    # Calculate tfidf scores for each term in query q
    Tfidf_query = compute_TfIdfQuery(q, idfDict)

    # add query vetor to tdm 
    for term in TDM.index:

        if term in Tfidf_query:
            Vector_q[term] = Tfidf_query[term]
        else:
            Vector_q[term] = 0

    TDM['query'] = Vector_q.values()

    # calculate cosine:
    for col in TDM:
        scores[col] = vec_cosine(TDM['query'], TDM[col])

    del scores["query"]

    return scores

### 2.2 cosine_scores()

# **Note that this is already a more efficient version of cosine computation.

def cosine_scores(q, D, I, L, idfDict):
    """Using inverted index. Code see lecture 4, slide 28. QUESTION: What is meant with 
    normalization: document length or vector norm? """

    tf_idf_query = dict()     # holds tf_idf scores for each term in q
    scores = dict()           # holds scores cos(q, d) for each document d in D

    # Initialize scores with zeros
    for idx in D.keys():
        key_ = idx
        value_ = 0
        scores.update({key_: value_})

    tf_idf_query = compute_TfIdfQuery(q, idfDict)
    
    
    # CORE IMPLEMENTATION OF COSINE Similarity 
    # Follows the pseudo-code given in Lecture 4, slide 28.
    
    # for each query term t
    for term in tf_idf_query.keys():

        # assign tf-idf of term t in query q to w_tq
        w_tq = tf_idf_query[term]

        # if term is in invertedIndex
        if term in I.keys():

            # fetch posting list for term t
            posting_list = I[term]

            # for each tuple (docID, tf-idf of term t in document d)
            for tuple_ in posting_list:

                docID = tuple_[0]
                w_td = tuple_[1]

                scores[docID] += (w_td * w_tq)

        else:
            continue

    # normalize scores with document length.
    for doc in scores.keys():
        if scores[doc] > 0:
            scores[doc] /= L[doc]

    return scores

### 2.2a cosine_scores_postingMerge()

def cosine_scores_postingMerge(q, D, I, L, idfDict, postingMergeIntersection=False):
    
    """Using inverted index. Code see lecture 4, slide 28. QUESTION: What is meant with 
    normalization: document length or vector norm? """

    tf_idf_query = dict()     # holds tf_idf scores for each term in q
    scores = dict()           # holds scores cos(q, d) for each document d in D

    # Initialize scores with zeros
    for idx in D.keys():
        key_ = idx
        value_ = 0
        scores.update({key_: value_})

    tf_idf_query = compute_TfIdfQuery(q, idfDict)

    # if posting merge: compute intersection of postings lists
    if postingMergeIntersection:
        relevant_document_ids = intersection(
            [set([t[0] for t in I[term]]) for term in tf_idf_query.keys()])
    
    # check if set of relevant documents is empty!
    if relevant_document_ids:
        
        # for each query term t
        for term in tf_idf_query.keys():

            # assign tf-idf of term t in query q to w_tq
            w_tq = tf_idf_query[term]

            # if term is in invertedIndex
            if term in I.keys():

                # fetch posting list for term t
                posting_list = I[term]

                # for each tuple (docID, tf-idf of term t in document d)
                for tuple_ in posting_list:

                    # assigne document id to docID
                    docID = tuple_[0]

                    # check if docID is in set of relevant_document_ids
                    if docID in relevant_document_ids:

                        w_td = tuple_[1]
                        
                        scores[docID] += (w_td * w_tq)

                        #if docID in scores:
                         #   scores[docID] += (w_td * w_tq)

                        #else:
                         #   scores[docID] = (w_td * w_tq)
                    else:
                        continue

            else:
                continue
    
    
    # normalize scores with document length.
    for doc in scores.keys():
        if scores[doc] != 0:
            scores[doc] /= L[doc]
    
    
    return scores

### 2.2b cosine_scores_pre_cluster()

def cosine_scores_pre_cluster(q, D, I, L, idfDict, preClusterDict, k=30,):
    
    """Own pre_clustering implementation, Lecture 4 slide P34, P35"""

    tf_idf_query = dict()        # holds tf_idf scores for each term in q
    scores_leaders = dict()      # holds scores cos(q, d) for each document d in list of cluster leaders!
    scores = dict()              # holds scores cos(q, d) for each document d in D

    clusters = preClusterDict
    
    # Initialize scores with zeros!
    for idx in D.keys():
        key_ = idx
        value_ = 0
        scores.update({key_: value_})

    # Initialize scores_leaders with zeros!
    for idx in clusters.keys():
        key_ = idx
        value_ = 0
        scores_leaders.update({key_: value_})

    tf_idf_query = compute_TfIdfQuery(q, idfDict)
    
    
    ### STEP 1: Compute cosine between query and cluster leaders!
    leader_documents_ids = clusters.keys()

    # for each query term t
    for term in tf_idf_query.keys():

       # assign tf-idf of term t in query q to w_tq
        w_tq = tf_idf_query[term]

        # if term is in invertedIndex
        if term in I.keys():
            
            # fetch posting list for term t
            posting_list = I[term]
            
            # for each tuple (docID, tf-idf of term t in document d)
            for tuple_ in posting_list:
                
                # assigne document id to docID
                docID = tuple_[0]

                # check if docID is in set of cluster leaders
                if docID in leader_documents_ids:

                    w_td = tuple_[1]
                    
                    scores_leaders[docID] += w_td * w_tq

                
                else:
                    continue
        else:

            continue
    
    
    # normalize scores with document length.
    for doc in scores_leaders.keys():
        if scores_leaders[doc] != 0:
            scores_leaders[doc] /= L[doc]
    
    
    # rank cosine scores of leaders
    ranking = list()

    for (id_, score_) in scores_leaders.items():
        tuple_ = (id_, score_)
        ranking.append(tuple_)

    ranking = sorted(ranking, key=getKeyForSorting, reverse=True)
    
    
    ### STEP 2: Compute cosine between query and all documents in cluster of max cluster leader!
    ### If we retrieve less than k=30 documents go to the cluster of the second cluster leader
    
    ranking_list_index = 0                # start with cluster leader max
    counter = 0                           # count number of docs found in cluster!
    
    while (counter < k and ranking_list_index < len(ranking)):

        relevant_document_ids = clusters[ranking[ranking_list_index][0]]

        for term in tf_idf_query.keys():

            # assign tf-idf of term t in query q to w_tq
            w_tq = tf_idf_query[term]

            # if term is in invertedIndex
            if term in I.keys():
                
                # fetch posting list for term t
                posting_list = I[term]
                
                # for each tuple (docID, tf-idf of term t in document d)
                for tuple_ in posting_list:
                    
                    
                    # assigne document id to docID
                    docID = tuple_[0]
                    
                    # check if docID is in set of relevant_document_ids
                    if docID in relevant_document_ids:
                        
                        w_td = tuple_[1]
                        
                        scores[docID] += w_td * w_tq
                    
                        #if docID in scores:
                         #   scores[docID] += w_td * w_tq
                        #else:
                         #   scores[docID] = w_td * w_tq
                    
                    else:
                        continue
            else:
                continue

        # Normalize scores and count number of relevant (i.e non zero documents in cluster)     
        for doc in scores.keys():
            
            if doc in relevant_document_ids and scores[doc] != 0:
                counter += 1
                scores[doc] /= L[doc]

        ranking_list_index += 1         # update index to go to next cluster!
    
    
    # Return scores
    return scores

### 2.2c cosine_scores_tiered()

def cosine_scores_tiered(
    q,
    D,
    T,
    L,
    idfDict,
    k,
):
    """Own tiered index implementation, Lecture 4 slide P41, P42"""

    tf_idf_query = dict()  # holds tf_idf scores for each term in q
    scores = dict()  # holds scores cos(q, d) for each document d in D
    pl_tier2 = []

    # Initialize scores with zeros!

    for idx in D.keys():
        key_ = idx
        value_ = 0
        scores.update({key_: value_})

    tf_idf_query = compute_TfIdfQuery(q, idfDict)

    # ## THIS IS THE CORE IMPLEMENTATION OF COSINE CALCULATION ###
    # for each query term t
    for term in tf_idf_query.keys():

        w_tq = tf_idf_query[term]

        if term in T.keys():

            posting_list = T[term]

            pl_tier1 = posting_list[0]  # tier 1"

            for tuple_ in pl_tier1:  # check the tuple in tier1

                docID = tuple_[0]

                w_td = tuple_[1]

                if docID in scores:
                    
                    scores[docID] += w_td * w_tq
                    
                else:

                    scores[docID] = w_td * w_tq
        else:

            continue

    counter = 0
    for doc in scores.keys():
        if scores[doc] > 0:
            counter += 1

    if counter < k:

        for term in tf_idf_query.keys():

            w_tq = tf_idf_query[term]

        # if term is in invertedIndex
            if term in T.keys():

                # fetch posting list for term t
                posting_list = T[term]

                pl_tier2 = posting_list[1]  # tier 2"
                for tuple_ in pl_tier2:  # check the tuple in tier2
                    docID = tuple_[0]
                    w_td = tuple_[1]

                    if docID in scores:
                        scores[docID] += w_td * w_tq
                    else:
                        scores[docID] = w_td * w_tq
            else:

                continue

    # normalize scores with document length.
    for doc in scores.keys():
        if scores[doc] > 0:
            #counter += 1
            scores[doc] /= L[doc]

    return scores

### 2.3 top_k_retrieval()

# **Idea:** This will be the main function of our Retrieval system.
# Based on strategy = ['vanilla', 'standard', 'intersection', 'preClustering', 'tiered']
# it will call a different fucntion to compute cosine scores
def getKeyForSorting(item):
    return item[1]


def top_k_retrieval(
    q,
    D,
    k,
    strategy='standard',
    idfDict=None,
    invertedIdx=None,
    lengthIdx=None,
    preClusterDict=None,
    tieredIdx=None,
    TDM=None,
    show_documents=False,
    return_results=False,
    print_scores=True,
    return_speed=False,
    ):
    
    """Input: Precomputed cosine scores, or query q and document collection D. Compute cosine scores between
    document d and query q. Convert dictionary entries to list and sort according to 
    cosine score. Return only top k entries with highest similarity between q and d."""

    t0 = time()

    # compute cosine scores:
    if strategy == 'intersection':
        scores = cosine_scores_postingMerge(q, D, invertedIdx, lengthIdx,
                idfDict, postingMergeIntersection=True)
    elif strategy == 'vanilla':
        scores = vanilla_cosine(q, TDM, idfDict)
    elif strategy == 'preclustering':
        scores = cosine_scores_pre_cluster(
            q,
            D,
            invertedIdx,
            lengthIdx,
            idfDict,
            preClusterDict,
            k=30,
            )
    elif strategy == 'tiered':
        scores = cosine_scores_tiered(
            q=q,
            D=D,
            T=tieredIdx,
            L=lengthIdx,
            idfDict=idfDict,
            k=30,
            )
    else:
        scores = cosine_scores(q, D, invertedIdx, lengthIdx, idfDict)

        
    # rank the returned scores
    ranking = list()

    for (id_, score_) in scores.items():
        tuple_ = (id_, score_)
        ranking.append(tuple_)

    ranking = sorted(ranking, key=getKeyForSorting, reverse=True)
    
    # append only topK scores to topK_list
    topK = list()
    for i in range(k):
        topK.append(ranking[i])
    
    # compute retrieval time
    retrieval_speed = time() - t0
    
    # if true print retrieval time and cosine scores summary
    if print_scores:
        print('=' * 50)
        print('Retrieval time ca. {:.8f} seconds.'.format(retrieval_speed))
        print('Highest cosine similarity:')
        for (doc_name, cos_) in topK:
            print("\t" + doc_name + " : {:.5f}".format(cos_))

        print('=' * 50)
        print()  # new line
    
    # if true show documents to user!
    if show_documents:
        for (doc_name, cos_) in topK:
            print(D[doc_name])
            print()
    
    # if true return scores + retrieval time 
    if return_results and return_speed:
        return (topK, retrieval_speed)
    
    # if true return scores
    if return_results:
        return topK



# 3. *Efficient Retrieval*

### 3.1 intersection()

def intersection(sets):
    """Returns the intersection of all sets in the list sets. Requires
   that the list sets contains at least one element, otherwise it
   raises an error."""

    return reduce(set.intersection, [s for s in sets])

### 3.2 pre_clustering()

def pre_cluster(D, verbose=True):
    
    """Some docString here"""

#     cluster = {'leader1': [doc1, doc2,,,],
#                'leader2': [doc4, doc5,,,]}

    t0 = time()

    cluster = dict()

   # number of leaders is the square root of number of docs
    num_leaders = math.trunc(math.sqrt(len(D)))

   # randomly choose leaders
    leaders = np.random.choice(list(D.keys()), size=num_leaders,
                               replace=False)

   # get list of non-leaders
    non_leaders = []
    for key in D.keys():
        if key not in leaders:
            non_leaders.append(key)

   # delete non_leaders from D
    D_leaders = copy.deepcopy(D)
    for non_leader in non_leaders:
        if non_leader in D_leaders.keys():
            del D_leaders[non_leader]

   # compute indexing based on document leader collection
    idfs_leaders = compute_Idf(D_leaders, verbose=False)
    tfidfs_leaders = compute_TfIdf(D_leaders, idfs_leaders,
                                   verbose=False)
    inverted_index_leaders = construct_invertedIndex(D_leaders,
                                                     idfs_leaders, tfidfs_leaders, verbose=False)
    doc_lengths_leaders = construct_docLengthDict(D_leaders,
                                                  tfidfs_leaders, verbose=False)

   # assign non_leaders to leader
   # (treat each non_leader document d as query)
    for doc in non_leaders:

        similarity = cosine_scores(q=D[doc], D=D,
                                   I=inverted_index_leaders,
                                   L=doc_lengths_leaders,
                                   idfDict=idfs_leaders)

        if list(max(zip(similarity.values(), similarity.keys())))[1] \
                in cluster:
            cluster[list(max(zip(similarity.values(),
                                 similarity.keys())))[1]].append(doc)
        else:
            cluster[list(max(zip(similarity.values(),
                                 similarity.keys())))[1]] = [doc]

    if verbose:
        print('Preclustering done in {:.4f}s.'.format(time() - t0))

    return cluster



# *Evaluation*

def evaluate_pAtRank(y_pred, y_true, atRank=10):
    
    """Calculate precision at Rank k"""

    retrieved_docs = list()  # holds relevant docID
    rel_01 = list()

    for i in range(len(y_pred)):
        retrieved_docs.append(y_pred[i][0])

    # convert into 0,1 vector
    for i in range(len(retrieved_docs)):
        if retrieved_docs[i] in y_true:
            rel_01.append(1)
        else:
            rel_01.append(0)

    # precision @Rank (by default p@10)
    precisionAtRank = (sum(rel_01)) / atRank

    return precisionAtRank

def evaluate_AveragePrecision(y_pred, y_true, k=None):
    
    """Some docString here"""

    retrieved_docs = list()             # holds relevant docID
    rel_01 = list()
    Pk_scores = list()                  # holds the precision scores at rank K

    for i in range(len(y_pred)):
        retrieved_docs.append(y_pred[i][0])

    # convert into 0,1 vector
    for i in range(len(retrieved_docs)):
        if retrieved_docs[i] in y_true:
            rel_01.append(1)
        else:
            rel_01.append(0)

    # compute precision at rank k
    for i in range(len(rel_01)):
        if rel_01[i] == 1:
            pAtK = sum(rel_01[:(i+1)]) / (i+1)
            Pk_scores.append(pAtK)
        else:
            continue

    # compute average precision across ranks
    if k is None:
        k = len(y_true)
        
    avgP = sum(Pk_scores) / k

    return avgP

def evaluate_nDCG(y_pred, y_true, variant="raw_scores"):
    
    """Some docString here"""

    rel_012 = list()
    retrieved_docIds = list()

    # extract IDs of retrieved documents
    for i in range(len(y_pred)):
        retrieved_docIds.append(y_pred[i][0])

    # convert gold standard into dictionary that is easier to use
    y_trueDict = dict()
    for i in range(len(y_true)):
        key_ = y_true[i][0]
        value_ = y_true[i][1]
        y_trueDict.update({key_: value_})

    # create vector that denotes the true relevance score for every retrieved document
    for i in range(len(retrieved_docIds)):
        if retrieved_docIds[i] in y_trueDict.keys():
            if y_trueDict[retrieved_docIds[i]] == 2:
                rel_012.append(2)
            if y_trueDict[retrieved_docIds[i]] == 1:
                rel_012.append(1)
        else:
            rel_012.append(0)
    
    if variant == "raw_scores":
        DCG = 0.0

        for idx in range(len(rel_012)):
            i = idx + 1
            score_ = (rel_012[idx]) / (np.log2(i + 1))
            DCG += score_

        # Calculate IDCG
        ideal_ranking = sorted(list(y_trueDict.values()), reverse=True)

        IDCG = 0.0
        for idx in range(len(ideal_ranking)):
            i = idx + 1
            score_ = (ideal_ranking[idx]) / (np.log2(i + 1))
            IDCG += score_

        # Noralized Discounted Cumulative Gain (nDCG)
        if IDCG == 0:
            #print("Flag Error!")
            nDCG = 0
        else:
            nDCG = DCG / IDCG    
    
    
    if variant == "power":
        # Calcualte second variant of DCG
        DCG2 = 0.0

        for idx in range(len(rel_012)):
            i = idx + 1
            score_ = (np.power(2, rel_012[idx]) - 1) / (np.log2(i + 1))
            DCG2 += score_

        # Calculate IDCG
        ideal_ranking = sorted(list(y_trueDict.values()), reverse=True)

        IDCG2 = 0.0
        for idx in range(len(ideal_ranking)):
            i = idx + 1
            score_ = (np.power(2, ideal_ranking[idx]) - 1) / (np.log2(i + 1))
            IDCG2 += score_

        # Noralized Discounted Cumulative Gain (nDCG)
        if IDCG2 == 0:
            nDCG = 0
        else:
            nDCG = DCG2 / IDCG2

    return nDCG


