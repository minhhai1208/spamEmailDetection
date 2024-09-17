import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Input, Conv1D, MaxPool1D,GlobalMaxPool1D, Embedding, BatchNormalization, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2


# read data
df = pd.read_csv("D:\\spam.csv", encoding='ISO-8859-1')
print(df.columns)

# clean data
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

# create new label
df["label"] = df["label"].map({"ham": 0, "spam":1})
print(df.head())

# check null and duplicate value
print(df.isna().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

# check balance of the data set
plt.hist(df[df["label"] == 0]["label"], bins=2, label="No spam")
plt.hist(df[df["label"] == 1]["label"], bins=2, label="Spam")
plt.legend()
plt.savefig("Before using SMOTE ")
plt.close()

# remove stop word
stopwords = set(stopwords.words('english'))
def removeStopWords(text):
    words = word_tokenize(text)
    filterd_word = [token for token in words if token.lower() not in stopwords ]
    return " ".join(filterd_word)

# tokenizes the cleaned text (converting it into sequences of numbers),
df["text"] = df["text"].apply(removeStopWords)
print(df["text"].head())
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df["text"].values)
sequence_train = tokenizer.texts_to_sequences(df["text"].values)

totalUniqueToken = len(tokenizer.word_index)
print(totalUniqueToken)

# pad sequence 
data_train = pad_sequences(sequence_train)
print(df.shape)
print(data_train.shape)

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(data_train, df["label"])
T = X_resampled.shape[1]

# Check class distribution after SMOTE
plt.hist(y_resampled[y_resampled == 0], bins=2, label="No spam")
plt.hist(y_resampled[y_resampled == 1], bins=2, label="Spam")
plt.legend()
plt.savefig("After using SMOTE.png")
plt.close()

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, stratify=y_resampled.values)
# build model
model = Sequential()

model.add(Embedding(totalUniqueToken+1, 30))
model.add(Dropout(0.5))
model.add(Conv1D(filters=28, kernel_size = 10, activation = "relu", kernel_regularizer=l2(0.5)))
model.add(BatchNormalization())
model.add(MaxPool1D(2, strides = 2))

model.add(Conv1D(filters=28, kernel_size = 10, activation = "relu", kernel_regularizer=l2(0.5)))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())

model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
CNN_model = model.fit(x=x_train, y=y_train, epochs=4, validation_data=(x_test, y_test))
print(CNN_model)

epochs = [i for i in range(4)]
fig, ax = plt.subplots(1, 2)
train_acc = CNN_model.history["accuracy"]
train_loss = CNN_model.history["loss"]
val_acc = CNN_model.history["val_accuracy"]
val_loss = CNN_model.history["val_loss"]
fig.set_size_inches(16, 8)

ax[0].plot(epochs, train_acc , label = "Training Accuracy")
ax[0].plot(epochs, val_acc, label = "Validaton Accuracy")
ax[0].set_title("Training and Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, label = "Training Loss")
ax[1].plot(epochs, val_loss, label= "Validation Loss")
ax[1].set_title("Testing Accuracy and Loss")
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
plt.savefig("Loss and Accuracy per epoch.png")