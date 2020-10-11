<h1>Next Word Predictor</h1>
<h3>Description</h3>
<p>Next Word Predictor is a project developed with an aim to determine the next word that comes after a particular word is typed. This model will consider the word in a sequence
and predict the next word. It is widely used in the applications we use widely like e-mail,social networking apps, search engines like google etc. 
The methods of natural language processing, language modeling, and deep learning will be used to build the model.The deep learning model will be built using LSTM( Long Short Term Memory).
</p>
<h3>Algorithms used</h3>
<h5>RNN-LSTM(Recurrent neural network-Long Short Term Memory):</h5>
<p>Long short-term memory is an artificial recurrent neural network architecture used in the field of deep learning. LSTM uses the concept of feedback connections.
LSTM model uses Deep learning with a network of artificial “cells” that manage memory, making them better suited for text prediction than traditional neural 
networks and other models.LSTM tackles the long-term dependency problem because it has memory cells to remember the previous context. Therefore LSTM approach is the best approach 
to build the model.</p>
<h3>Functioning of the code</h3>
<table>
<tr><th>Import all the necessary libraries</th></tr>
<tr><td>from numpy import array<br>
from keras.preprocessing.text import Tokenizer<br>
from keras.utils import to_categorical<br>
from keras.models import Sequential<br>
from keras.layers import Dense<br>
from keras.layers import LSTM<br>
from keras.layers import Embedding<br></td></tr>
<tr><th>Provide the dataset to perform the operation of next word prediction</th></tr>
<tr><td>dataset="""Hello this is my project on next word predictor. I am working as an intern in DevIncept. This project is developed based on RNN-LSTM
. I am a AI enthusiast and keen to work on many more projects."""</td></tr>
<tr><th>Creating an instance of tokenizer and encoding the dataset</th></tr>
<tr><td>tokenizer=Tokenizer()<br>
tokenizer.fit_on_texts([dataset])<br>
encode=tokenizer.texts_to_sequences([dataset])[0]</td></tr>
<tr><td><b>Explanation:</b> <p>Tokenizer is used to encode the input dataset. Tokenizer is imported from <i> keras.preprocessing.text</i>.Basically encoded to integer form with 
Tokenizer. <i>tokenizer.fit_on_texts([dataset])</i> fits the data into tokenizer to convert into encoded data. <i>tokenizer.texts_to_sequences([dataset])[0]</i> converts the dataset
into sequence of encoded for i.e numercal form.</p></td></tr>
<tr><th>Finding the number of words in the given dataset</th></tr>
<tr><td>vocab=len(tokenizer.word_index) + 1</td></tr>
<tr><td><b>Explanation:</b> <p>To findout the number of words in the given dataset, use a method in tokenizer called "word_index", which basically finds the number of words. Initialize the size to vocab</p></td></tr>
<tr><th>Creating word sequences</th></tr>
<tr><td>sequences = list()<br>
for i in range(1,len(encode)):<br>
&nbsp;&nbsp;    sequence = encode[i-1:i+1]<br>
&nbsp;&nbsp;    sequences.append(sequence)</td></tr>
<tr><td><b>Explanation:</b><p> A sequence of words should be created to fit the model with one word as input and the other as output. We create a list of sequences possible with 
the dataset. <i>for i in range(1,len(encode)):</i> It will traverse from the first to end of the above encoded data. <i>sequence = encode[i-1:i+1]</i>: Creates a sequence
of a particular word and its succeding word.<i>sequences.append(sequence)</i>: It will create a list of sequences of words.</p></td></tr>

<tr><th>Splitting the sequences into input(x) and output(y)</th></tr>
<tr><td>sequences = array(sequences)<br>
x,y = sequences[:,0],sequences[:,1]</td></tr>
<tr><td><b>Explanation:</b><p>The sequences is converted into an array and the data is splitted into x and y as shown above that is the input word and the output word.</p></td></tr>
<tr><th>Converting 'y' into one hot encode outputs</th></tr>
<tr><td>y=to_categorical(y,num_classes=vocab)</td></tr>
<tr><td><b>Explanation:</b><p>One hot encoding should be applied on the output elements of 'y'. <i>y=to_categorical(y,num_classes=vocab)</i> requires two parameters i.e the variable
in which data is stored and the number of output classes i.e number of different classes, which is equal to the size of the dataset or number of words in the dataset.</p></td></tr>
<tr><th>Creating the model</th></tr>
<tr><td>mymodel=Sequential()<br>
mymodel.add(Embedding(vocab,10,input_length=1))<br>
mymodel.add(LSTM(50))<br>
mymodel.add(Dense(vocab,activation='softmax'))<br>
print(mymodel.summary())</td></tr>
<tr><td><b>Explanation:</b> <p>Create a sequential model called 'mymodel'. Our sequential model will have three layers: Embedding,LSTM and Dense. Embedding layer is imported from 
<i>from keras.layers</i>  It requires that the input data be integer encoded, so that each word is represented by a unique integer. It has three parameters: <i>input_dim: size of vocabulary
in the dataset,output_dim: the number of dimensions we wish to embed into, input_length</i>. LSTM(Long short term Memory) is an artificial recurrent neural network architecture used
in the field of deep learning. LSTM has feedback connections. '50' defines the number of neurons. Dense layer is a fully connected layer with two parameters: vocbulary size of the dataset
and activation. <i>activation='softmax'</i>: Softmax is used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution</p></td></tr>

<tr><th><Compiling the network</th></tr>
<tr><td>mymodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])</td></tr>
<tr><th>Training the model</th></tr>
<tr><td>mymodel.fit(x,y,epochs=100)</td></tr>
<tr><td><b>Explanation:</b><p>We have to train the model using the input and output data we have divided i.e x and y. <i>epochs=100</i> is used to separate training into distinct phases,
which is useful for logging and periodic evaluation.</p></td></tr>
<tr><th>Building the logic to print the predicted word</th></tr>
<tr><td>def getseq(mymodel,tokenizer,text,n_pred):<br>
&nbsp;&nbsp;    inp,result=text,text<br>
&nbsp;&nbsp    for _ in range(n_pred):<br>
&nbsp;&nbsp&nbsp;&nbsp        encode=tokenizer.texts_to_sequences([inp])[0]<br>
&nbsp;&nbsp&nbsp;&nbsp        encode=array(encode)<br>
&nbsp;&nbsp&nbsp;&nbsp&nbsp;&nbsp&nbsp;&nbsp        yt=mymodel.predict_classes(encode)<br>
&nbsp;&nbsp&nbsp;&nbsp        out=''<br>
&nbsp;&nbsp&nbsp;&nbsp        for word,index in tokenizer.word_index.items():<br>
&nbsp;&nbsp&nbsp;&nbsp&nbsp; if index==yt:<br>
&nbsp;&nbsp&nbsp;&nbsp&nbsp;&nbsp;out=word<br>
&nbsp;&nbsp&nbsp;&nbsp&nbsp;&nbsp;break<br>
&nbsp;&nbsp&nbsp;&nbsp inp,result=out,result+' '+out<br>
&nbsp;&nbsp  return result</td></tr>
<tr><td><b>Explanation:</b> <p><i>def getseq(mymodel,tokenizer,text,n_pred)</i> define a function with parameters mymodel : which defines the model we have created, tokenizer,
text: defines the word after which we want to predict its probable words, n_pred: number of predicted words we want to print after the given word.</p></td></tr>
<tr><td><p><i> for _ in range(n_pred):</i> iterates from 1 till the number od probable words we want to print. <i>encode=tokenizer.texts.to_sequences([inp])[0]</i>: Converts 
the input into sequence of encoded data using tokenizer. <i>array(encode)</i>: Converts the enocded data into array. <i>yt=model.predict_classes(encode)</i>:Predicts a word in the 
vocabulary/data.<i>for word,index in tokenizer.word_index.items():</i> iterates for every word and index to find the predicted output.<i> if index==yt:</i> It checks if index 
is equal to predicted output then out will be initialized with the word and break the loop.Now it will return the result which is the combination of 'text' and the predicted words.</p></td></tr></table>
<h3> Code for the above model</h3>
<table><tr><td>from numpy import array<br>
from keras.preprocessing.text import Tokenizer<br>
from keras.utils import to_categorical<br>
from keras.models import Sequential<br>
from keras.layers import Dense<br>
from keras.layers import LSTM<br>
from keras.layers import Embedding<br><br><br>

dataset="""Hello this is my project on next word predictor. I am working as an intern in DevIncept. This project is developed based on RNN-LSTM
. I am a AI enthusiast and keen to work on many more projects."""<br><br>

tokenizer=Tokenizer()<br>
tokenizer.fit_on_texts([dataset])<br>
encode=tokenizer.texts_to_sequences([dataset])[0]<br>
encode<br><br>

vocab=len(tokenizer.word_index) + 1<br>
print("size of vocab is %d" % vocab)<br><br>

sequences = list()<br>
for i in range(1,len(encode)):<br>
&nbsp;&nbsp;    sequence = encode[i-1:i+1]<br>
&nbsp;&nbsp;   sequences.append(sequence)<br>
&nbsp;&nbsp;   print("total sequences : %d" % len(sequences))<br><br>

y=to_categorical(y,num_classes=vocab)<br><br>

mymodel=Sequential()<br>
mymodel.add(Embedding(vocab,10,input_length=1))<br>
mymodel.add(LSTM(50))<br>
mymodel.add(Dense(vocab,activation='softmax'))<br>
print(mymodel.summary())<br><br>

mymodel.compile(loss='categorical_crossentropy',optimmizer='adam',metrics=['accuracy'])<br><br>

mymodel.fit(x,y,epochs=100)<br><br>

def getseq(mymodel,tokenizer,text,n_pred):<br>
&nbsp;&nbsp;    inp,result=text,text<br>
&nbsp;&nbsp;    for _ in range(n_pred):<br>
&nbsp;&nbsp;&nbsp;&nbsp;encode=tokenizer.texts_to_sequences([inp])[0]<br>
&nbsp;&nbsp;&nbsp;&nbsp;encode=array(encode)<br>
&nbsp;&nbsp;&nbsp;&nbsp;yt=mymodel.predict_classes(encode)<br>
&nbsp;&nbsp;&nbsp;&nbsp;        out=''<br>
&nbsp;&nbsp;&nbsp;&nbsp;        for word,index in tokenizer.word_index.items():<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if index==yt:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; out=word<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; break<br>
&nbsp;&nbsp;&nbsp;&nbsp; inp,result=out,result+' '+out<br>
&nbsp;&nbsp;  return result<br><br>

print(getseq(mymodel,tokenizer,'Hello',1))</td></tr></table>

<h3>Output</h3>
<img src=""/>



    





<tr><th>


