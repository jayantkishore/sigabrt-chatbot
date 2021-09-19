# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"
import json
from typing import Any, Text, Dict, List
import torch
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import os


pretrained_model= 'bert-base-nli-mean-tokens' # Refer: https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md
score_threshold = 0.80  # This confidence scores can be adjusted based on your need!!

# Custom Action
class ActionGetFAQAnswer(Action):

    def __init__(self):
        super(ActionGetFAQAnswer, self).__init__()
        #self.faq_data = json.load(open("./data/nlu/faq.json", "rt", encoding="utf-8"))
        self.sentence_embedding_choose(pretrained_model)
        #self.standard_questions_encoder = np.load("./data/standard_questions.npy")
        #self.standard_questions_encoder_len = np.load("./data/standard_questions_len.npy")
        #print(self.standard_questions_encoder.shape)

    def sentence_embedding_choose(self, pretrained_model='bert-base-nli-mean-tokens'):
        
        self.bc = SentenceTransformer(pretrained_model)
        

    def get_most_similar_standard_question_id(self, query_question):
        
        query_vector = torch.tensor(self.bc.encode([query_question])[0]).numpy()
        
        print("Question received at action engineer")
        score = np.sum((self.standard_questions_encoder * query_vector), axis=1) / (
                self.standard_questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
        top_id = np.argsort(score)[::-1][0]
        return top_id, score[top_id]

    def name(self) -> Text:
        return "action_get_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['text']
        intent = str(tracker.latest_message['intent']['name'])
        faq_path = "./data/nlu/json_files/{}.json".format(intent)
        self.faq_data = json.load(open(faq_path , "rt" ,  encoding="utf-8"))
        #print(query)
        ques_path = "./data/bert_encoding/stan_ques_{}.npy".format(intent)
        ques_len_path = "./data/bert_encoding/stan_ques_len_{}.npy".format(intent)

        self.standard_questions_encoder = np.load(ques_path)
        self.standard_questions_encoder_len = np.load(ques_len_path)

        most_similar_id, score = self.get_most_similar_standard_question_id(query)
        print("The question is matched with id:{} with score: {}".format(most_similar_id,score))
        if float(score) > score_threshold: # This confidence scores can be adjusted based on your need!!
            response = self.faq_data[most_similar_id]['a']
            dispatcher.utter_message(response)
            dispatcher.utter_message("Problem solved?")
        else:
            response = "Sorry, this question is beyond my ability..."
            dispatcher.utter_message(response)
            dispatcher.utter_message("Sorry, I can't answer your question. You can dial the manual service...")
        return []


def encode_standard_question(pretrained_model='bert-base-nli-mean-tokens'):
    """
    This will encode all the questions available in question database into sentence embedding. The result will be stored into numpy array for comparision purpose.
    """
    
    path = "./data/nlu/json_files"
    lt = os.listdir(path)
    bc = SentenceTransformer(pretrained_model)
    for a in lt:
        name = a.split(".")[0]
        my_path = path + "/" + a 
        data = json.load(open(my_path , "rt" ,  encoding="utf-8"))
        questions = [each['q'] for each in data]
        sqe = torch.tensor(bc.encode(questions)).numpy()
        my_path_q = "./data/bert_encoding/stan_ques_" + name 
        np.save(my_path_q, sqe)
        sqe_len = np.sqrt(np.sum(sqe * sqe, axis=1))
        my_path_q_len = "./data/bert_encoding/stan_ques_len_" + name 
        np.save(my_path_q_len, sqe_len)
    
encode_standard_question(pretrained_model)
# if __name__ == '__main__':
#     encode_standard_question(True)
