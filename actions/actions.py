# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"
import os
import json
from typing import Any, Text, Dict, List
import torch
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from rasa_sdk.events import SlotSet
from datetime import date, timedelta, time, datetime
import urllib.request
from pytz import timezone

pretrained_model = "bert-base-nli-mean-tokens"  # Refer: https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md
score_threshold = 0.80  # This confidence scores can be adjusted based on your need!!

# Custom Action
class ActionGetFAQAnswer(Action):
    def __init__(self):
        super(ActionGetFAQAnswer, self).__init__()
        # self.faq_data = json.load(open("./data/nlu/faq.json", "rt", encoding="utf-8"))
        self.sentence_embedding_choose(pretrained_model)
        # self.standard_questions_encoder = np.load("./data/standard_questions.npy")
        # self.standard_questions_encoder_len = np.load("./data/standard_questions_len.npy")
        # print(self.standard_questions_encoder.shape)

    def sentence_embedding_choose(self, pretrained_model="bert-base-nli-mean-tokens"):

        self.bc = SentenceTransformer(pretrained_model)

    def get_most_similar_standard_question_id(self, query_question):

        query_vector = torch.tensor(self.bc.encode([query_question])[0]).numpy()

        print("Question received at action engineer")
        score = np.sum((self.standard_questions_encoder * query_vector), axis=1) / (
            self.standard_questions_encoder_len
            * (np.sum(query_vector * query_vector) ** 0.5)
        )
        top_id = np.argsort(score)[::-1][0]
        return top_id, score[top_id]

    def name(self) -> Text:
        return "action_get_answer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        query = tracker.latest_message["text"]
        intent = str(tracker.latest_message["intent"]["name"])
        faq_path = "./data/nlu/json_files/{}.json".format(intent)
        self.faq_data = json.load(open(faq_path, "rt", encoding="utf-8"))
        # print(query)
        ques_path = "./data/bert_encoding/stan_ques_{}.npy".format(intent)
        ques_len_path = "./data/bert_encoding/stan_ques_len_{}.npy".format(intent)

        self.standard_questions_encoder = np.load(ques_path)
        self.standard_questions_encoder_len = np.load(ques_len_path)

        most_similar_id, score = self.get_most_similar_standard_question_id(query)
        print(
            "The question is matched with id:{} with score: {}".format(
                most_similar_id, score
            )
        )
        if (
            float(score) > score_threshold
        ):  # This confidence scores can be adjusted based on your need!!
            response = self.faq_data[most_similar_id]["a"]
            dispatcher.utter_message(response)
            dispatcher.utter_message("Problem solved?")
        else:
            response = "Sorry, this question is beyond my ability..."
            dispatcher.utter_message(response)
            dispatcher.utter_message(
                "Sorry, I can't answer your question. You can dial the manual service..."
            )
        return []


def encode_standard_question(pretrained_model="bert-base-nli-mean-tokens"):
    """
    This will encode all the questions available in question database into sentence embedding. The result will be stored into numpy array for comparision purpose.
    """

    path = "./data/nlu/json_files"
    lt = os.listdir(path)
    bc = SentenceTransformer(pretrained_model)
    for a in lt:
        name = a.split(".")[0]
        my_path = path + "/" + a
        data = json.load(open(my_path, "rt", encoding="utf-8"))
        questions = [each["q"] for each in data]
        sqe = torch.tensor(bc.encode(questions)).numpy()
        my_path_q = "./data/bert_encoding/stan_ques_" + name
        np.save(my_path_q, sqe)
        sqe_len = np.sqrt(np.sum(sqe * sqe, axis=1))
        my_path_q_len = "./data/bert_encoding/stan_ques_len_" + name
        np.save(my_path_q_len, sqe_len)


class Aux:
    def __init__(self):
        self.name = ""

    def getinfo(self, ent):
        if len(ent) == 0:
            return -1
        url = "https://ticker-2e1ica8b9.now.sh/keyword/"
        comp = ent[0]["value"].lower()
        url += comp
        u = urllib.request.urlopen(url)
        content = u.read()
        obj = json.loads(content)
        if obj == []:
            return -1
        return obj[0]["symbol"]


class ActionJK(Action):
    def name(self):
        return "action_jk"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        aux = Aux()
        tick = aux.getinfo(tracker.latest_message["entities"])
        if tick != -1:
            today = datetime.now(timezone("US/Eastern"))
            if today.strftime("%A") == "Sunday":
                today = today - timedelta(days=1)
                dispatcher.utter_message(
                    text="Today is Sunday at NYSE! Showing results for yesterday"
                )
            elif today.strftime("%A") == "Monday" and today.time() <= time(10, 00):
                today = today - timedelta(days=2)
                dispatcher.utter_message(
                    text="NYSE is not open yet! Showing results for day before yesterday"
                )
            yesterday = today - timedelta(days=1)
            tod = today.strftime("%Y-%m-%d")
            yest = yesterday.strftime("%Y-%m-%d")
            print(yest)
            print(tod)
            prefix = "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?apiKey=Gsgns_Tv2e5X9WmjIEzqzK1dvgX1FTpU".format(
                tick, yest, tod
            )
            url = prefix
            u = urllib.request.urlopen(url)
            content = u.read()
            obj = json.loads(content)
            print(obj)
            rsp = obj["results"][0]
            ans = ""
            ans += "Open : {}  ||  Close : {}  ||  Low : {}  || High : {} ".format(
                rsp["o"], rsp["c"], rsp["l"], rsp["h"]
            )
            dispatcher.utter_message(
                text="Yes sure! By the Grace of Jayant , I know about {} stocks!".format(
                    tick
                )
            )
            dispatcher.utter_message(text=ans)
            return [SlotSet("status", True), SlotSet("counter", 0)]
        else:
            try:
                cnt = tracker.get_slot("counter")
                cnt += 1
                return [SlotSet("status", False), SlotSet("counter", cnt)]
            except:
                return [SlotSet("status", False), SlotSet("counter", 1)]


class ActionJK2(Action):
    def name(self):
        return "action_jk2"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        if tracker.get_slot("status") == False:
            if tracker.get_slot("counter") <= 2:
                dispatcher.utter_message(text="Be specific!")
            else:
                dispatcher.utter_message(
                    text="Sorry I can not fetch the details for this company!"
                )
                dispatcher.utter_message(text="Could you be more specific please?")
                dispatcher.utter_message(
                    text="Or maybe, you could try searching for another stock?"
                )
        else:
            dispatcher.utter_message(text="Are you planning to buy this stock?")
        return []


encode_standard_question(pretrained_model)
# if __name__ == '__main__':
#     encode_standard_question(True)
