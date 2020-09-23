# retrieve-similar-questions

First train the model by running the following code

```
python information_retrieval.py
```

Then you have the trained model. 

Run the app.py 
```
flask run
```

Use this to receive similar questions 
```
curl --header "Content-Type: application/json" --request POST --data '{"question": "what is the best phone in the indian market?"}' http://127.0.0.1:5000/ 
```

You should recieve 3 similar questions to the original question you sent.
