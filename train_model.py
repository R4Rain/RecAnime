import random
import numpy as np
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

# download once
# =================================
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
# =================================

words = []
labels = []
documents = []
words_ignore = ["?", "!", "{", "}", ","]
data_file = open("data/intents.json").read()
intents = json.loads(data_file)
lemmatizer = WordNetLemmatizer()

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

# lemmatizer
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in words_ignore]
words = sorted(list(set(words)))
labels = sorted(list(set(labels)))

# Store it locally
pickle.dump(words, open("data/words.pkl", "wb"))
pickle.dump(labels, open("data/labels.pkl", "wb"))

# Create the training data
training = []
output_empty = [0] * len(labels)
for doc in documents:
    # initializing bag of words
    bag = []
    pattern_words = doc[0]
    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word found in current pattern
    # else 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle and turn it into np.array
random.shuffle(training)
training = np.array(training, dtype='object')

# create train and test lists. X -> patterns, y -> intents
X_train = list(training[:, 0])
y_train = list(training[:, 1])
print("Training data created!")

# Training
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons 
# and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation="softmax"))
model.summary()

# Compile model with Stochastic gradient descent
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) # type: ignore
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# fitting and saving the model
hist = model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=5, verbose=1) # type: ignore
model.save("chatbot_model.h5", hist)
print("Model created!")