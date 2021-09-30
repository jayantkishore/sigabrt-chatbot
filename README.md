# SIGA Chatbot

Chatbot Server based on [Rasa 2.8.2](https://github.com/RasaHQ).

## Installation

Use the latest version of package manager [pip](https://pip.pypa.io/en/stable/) for installation.

```bash
pip install -r requirements.txt
```

## Usage
>The pre-trained models are stored in /models directory
```python
#train the assistant
rasa train
# Run the action server 
rasa run actions 

# Run the main chatbot server 
rasa run
```
* Action server listens on port 5055 by default
* Main server by default listens on port 5005

**API Endpoint**
> POST request can be sent to {server_base_url}/webhooks/rest/webhook in the following formats

```bash
$ curl -d '{"sender":"your_user_id_here", "message":"Hi"}' -H 'Content-Type: application/json' -X POST {server_base_url}/webhooks/rest/webhook
```
