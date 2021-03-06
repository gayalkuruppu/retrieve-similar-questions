# retrieve-similar-questions

To install conda if you do not have anaconda installed

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```
To install for 32 bits version remove "_64" and for python2 change anaconda3 to anaconda2

Create a conda virtual environment and install the dependencies as below

```
conda create -n quora python=3.6
pip install -r requirements.txt
conda activate quora
```


First train the model by running the following code

```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gunzip "GoogleNews-vectors-negative300.bin.gz"
mkdir clusters

python information_retrieval.py
```

Now you have the trained model. Run the API.

Run this code in terminal and copy the response to the clipboard
```
which python
```

Now run the app.py 
```
sudo <which python response> app.py
```

Use this in the client to receive similar questions 
```
curl --header "Content-Type: application/json" --request POST --data '{"question": "what is the best phone in the indian market?"}' http://127.0.0.1:5000/ 
```

You should recieve 3 similar questions to the original question you sent.
