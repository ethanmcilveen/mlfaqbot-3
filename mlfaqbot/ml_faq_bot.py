"""
Author: Ethan McIlveen, February 2023, 000858082
uses general format from faq_bot_skeleton provided and used in previous assignment, credit to Sam Scott, Mohawk College

script that implements document vectorization, importing pickled classifiers and vectorizers and using gpt3 to construct a ML-based FAQ bot

"""
##Imports
import math
import openai
from file_input import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def load_FAQ_data():
    """This method returns a list of questions and answers. This also loads the .joblib files for the corpus written, the MLPClassifier and the vectorizer."""

    #load questions, answers, and regex expressions
    questions = file_input("faq_questions.txt")
    answers = file_input("faq_answers.txt")
    from joblib import load
    clf = load('example_corpus.joblib')
    otherVectorizer = load('vectorizer.joblib')

    # Create the CountVectorizer and fit it with the questions
    vectorizer = CountVectorizer(ngram_range=(1,3))
    vectorizer.fit(questions)

    return questions, answers, clf, otherVectorizer, vectorizer

def understand(utterance):
    """This method processes an utterance to determine which intent it
    matches. The response is returned as a string """

    global questions #declare the global variables
    global answers
    global clf
    global otherVectorizer
    global vectorizer

    if utterance == "help": #if the user wants help, show them all the questions they can ask, return and dont proceed down the pipeline
        response = "Here is a list of questions you can ask me: \n"
        for question in questions:
            response = response + question + "\n" #return the list of questions
        return response
##first step of pipeline
    new_vector = vectorizer.transform([utterance])
    vectors = vectorizer.transform(questions)
    similarities = cosine_similarity(new_vector, vectors)[0]
    most_similar_index = similarities.argmax() #find the question with the highest similarity score (highest since using cosine similarity)
    most_similar_question = questions[most_similar_index]

    most_similar_index_answer = math.floor(most_similar_index / 3) #find an appropriate answer
    best_score = similarities[most_similar_index]
    print(best_score)

    if best_score > 0.75: #if the question is similair enough, print the answer from the list of answers
        response = answers[most_similar_index_answer]
    else:
##second step of pipeline
        new_vector = otherVectorizer.transform([utterance])#vectorize for determining user intent (greeting, farewell, thankful)
        prediction = clf.predict(new_vector)
        checkIntent = prediction[0]

        if checkIntent == 0:#send appropriate answer based on prediction
            response = "Hi there, how can I help you?"
        elif checkIntent == 1:
            response = "Bye for now!"
        elif checkIntent == 2:
            response = "You're very welcome!"
        else: #if it doesn't match, continue down the pipeline to a gpt3 response
            response = "Sorry, I can't help you with that."

    return response

##load the data
questions, answers, clf, otherVectorizer, vectorizer = load_FAQ_data()