# SIGA Voicebot

An interactive voice bot for user engagement powered by [Rasa](https://rasa.com/) for all your financial and [Flipkart](https://www.flipkart.com/) related queries. 

## Installation and Usage
Voicebot architecture is divided into three main components
1. **UI/UX Layer** powered by [React](https://reactjs.org/) 
>[Installation and usage](www.google.com)
2. **Backend Server** powered by [Django Rest Framework](https://www.django-rest-framework.org/) 
>[Installation and usage](www.google.com)
3. **Chatbot Server** fueled by [Rasa](https://rasa.com/) 
>[Installation and usage](www.google.com)

## Deployment
All the components are deployed on [Azure](https://azure.microsoft.com/en-in/) cloud platforms and VM.
You can access SIGA voice bot by clicking [here](https://nice-island-04efa0100.azurestaticapps.net/).

## Working of the Bot
### What is Rasa?
>[Rasa](https://rasa.com/) is an open-source machine learning framework for automated text and voice-based conversations. Understand messages, hold conversations, and connect to messaging channels and APIs. Rasa documentation [here](https://rasa.com/).



### Components
Rasa comes up with 2 components —
>***Rasa NLU*** — for natural language understanding (NLU) which does the classification of intent and extract the entity from the user input and helps bot to understand what the user is saying.

>***Rasa Core*** — a chatbot framework with machine learning-based dialogue management that takes the structured input from the NLU and predicts the next best action using a probabilistic model.


### Overview of the files

`data/nlu/nlu.yml` - contains NLU training data

`data/nlu/rules.yml` - contains rules training data

`data/stories/stories.yml` - contains stories training data

`actions.py` - contains custom action/api code

`domain.yml` - the domain file, including bot response templates

`config.yml` - training configurations for the NLU pipeline and policy ensemble

### Natural Language Understanding (NLU)
For intent classification and entity extraction we used the following pipeline:

**Language Models**
>The following components load pre-trained models of word vectors in the pipeline.

* [SpacyNLP](https://spacy.io) : [en_core_web_md](https://spacy.io/models/en#en_core_web_md)

**Tokenizers**
>Tokenizers split the receivedtext into tokens. We used the following in our pipeline.

* [SpacyTokenizer](https://spacy.io/api/tokenizer)

**Featurizers**
>Featurizers transform the words into meaningful numbers (or vectors) that can be fed to the training algorithm.
* [SpacyFeaturizer](https://rasa.com/docs/rasa/components/#spacyfeaturizer)

* [RegexFeaturizer](https://rasa.com/docs/rasa/components/#regexfeaturizer)

**Intent Classifiers and Entity Extractors**
>Intent classifiers assign one of the intents defined in the domain file to incoming user messages.
* [Dual Intent Entity Transformer (DIET) Classifier](https://rasa.com/docs/rasa/components/#dietclassifier)
* [NLU Fallback Classifier](https://rasa.com/docs/rasa/components/#fallbackclassifier)
* [RegexEntityExtractor](RegexEntityExtractor)
* [EntitySynonymMapper](https://rasa.com/docs/rasa/components/#entitysynonymmapper)

### Core Policies
>Policies to decide which action to take at each step in a conversation.
We used the following configuration with fine tuning of parameters:
1. Machine Learning-Based Policies

* **[Transformer Embedding Dialogue (TED) Policy](https://arxiv.org/abs/1910.00486)**

* **[Memoization Policy](https://rasa.com/docs/rasa/policies#memoization-policy)**

2. Rule-based Policies
* **[Rule Policy](https://rasa.com/docs/rasa/policies#rule-policy)**

### Domain-Specific Training
> We have scrapped common questions and answers according to our domain-specific needs from a wide range of websites that make the knowledge base of our bot. SIGA will intelligently answer user questions based on this knowledge base using the following model:

**[Sentence Transformers: Multilingual Sentence, Paragraph, and Image Embeddings using BERT](https://github.com/UKPLab/sentence-transformers)** 
* provides state-of-the-art pre-trained models which are used to compute sentence embedding. Later, sentence embeddings are used to calculate the similarity between the user's question and the questions in the knowledge base.

### Third-Party APIs Used

> Some third-party APIs have been leveraged to provide a better user experience.

* News Api: [CNBC News API](https://rapidapi.com/apidojo/api/cnbc)
* Stock Market API: [Polygon.io - Stock Market Data APIs](https://polygon.io/)
* Dummy APIs (self-created) for Credit Card, and Flipkart Orders 




