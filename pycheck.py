# -*- coding: utf-8 -*-



import cv2
import numpy as np
import streamlit as st
import pandas as pd
import speech_recognition as sr
import pandas as pd
import numpy as np
import os
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import argparse
from PIL import Image
filename="C:/Py-ML/sp.png"
img = cv2.imread(filename, 1)
image = np.array([img])
selectbox=st.sidebar.radio("Select Your Choice", ("Speech Summarization","Speech Translation","Gender Detection"))
#st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;"></p>', unsafe_allow_html=True)
#new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Speech Recognition</p>'
st.markdown("<h5 style='text-align: center; color:green; font-weight: bold;border: 3px solid #e6e9ef;border-radius: 0.4rem;font-style: italic;font-size:50px;'>Speech Recognition APP", unsafe_allow_html=True)
#st.markdown(new_title, unsafe_allow_html=True)





r=sr.Recognizer()
mic=sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)
    audio=r.listen(source)
    enable_automatic_punctuation=True
    #st.write(r.recognize_google(audio))
    m=r.recognize_google(audio)
    print(m)
    with open('speech.wav','wb') as f:
        f.write(audio.get_wav_data())
    m=r.recognize_google(audio)
    print(m)
file = "speech.wav"
words=m

df = pd.read_csv("C:/Py-ML/data/balanced-all.csv")
df.head()

df.tail()
# get total samples
n_samples = len(df)
# get total male samples
n_male_samples = len(df[df['gender'] == 'male'])
# get total female samples
n_female_samples = len(df[df['gender'] == 'female'])
print("Total samples:", n_samples)
print("Total male samples:", n_male_samples)
print("Total female samples:", n_female_samples)

label2int = {
    "male": 1,
    "female": 0
}

label2int = {
    "male": 1,
    "female": 0
}


# def load_data(vector_length=128):
#     """A function to load gender recognition dataset from `data` folder
#     After the second run, this will load from results/features.npy and results/labels.npy files
#     as it is much faster!"""
#     # make sure results folder exists
#     if not os.path.isdir("results"):
#         os.mkdir("results")
#     # if features & labels already loaded individually and bundled, load them from there instead
#     if os.path.isfile("C:/Py-ML/data/results/features.npy") and os.path.isfile("C:/Py-ML/data/results/labels.npy"):
#         X = np.load("C:/Py-ML/data/results/features.npy")
#         y = np.load("C:/Py-ML/data/results/labels.npy")
#         return X, y
#     # read dataframe
#     df = pd.read_csv("C:/Py-ML/data/balanced-all.csv")
#     # get total samples
#     n_samples = len(df)
#     # get total male samples
#     n_male_samples = len(df[df['gender'] == 'male'])
#     # get total female samples
#     n_female_samples = len(df[df['gender'] == 'female'])
#     print("Total samples:", n_samples)
#     print("Total male samples:", n_male_samples)
#     print("Total female samples:", n_female_samples)
#     # initialize an empty array for all audio features
#     X = np.zeros((n_samples, vector_length))
#     # initialize an empty array for all audio labels (1 for male and 0 for female)
#     y = np.zeros((n_samples, 1))
   
#     for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
#         print(filename)
#         features = np.load(filename)
        
#         print(df['filename'])
#         X[i] = features
#         y[i] = label2int[gender]
#     # save the audio features and labels into files
#     # so we won't load each one of them next run
#     np.save("results/features", X)
#     np.save("results/labels", y)
#     return X, y

# def split_data(X, y, test_size=0.1, valid_size=0.1):
#     # split training set and testing set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
#     # split training set and validation set
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
#     # return a dictionary of values
#     return {
#         "X_train": X_train,
#         "X_valid": X_valid,
#         "X_test": X_test,
#         "y_train": y_train,
#         "y_valid": y_valid,
#         "y_test": y_test
#     }

# X, y = load_data()
#  # split the data into training, validation and testing sets
# data = split_data(X, y, test_size=0.1, valid_size=0.1)

# def create_model(vector_length=128):
#     model = Sequential()
#     model.add(Dense(256, input_shape=(vector_length,)))
#     model.add(Dropout(0.3))
#     model.add(Dense(256, activation="relu"))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation="relu"))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation="relu"))
#     model.add(Dropout(0.3))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dropout(0.3))
#      # one output neuron with sigmoid activation function, 0 means female, 1 means male
#     model.add(Dense(1, activation="sigmoid"))
#      # using binary crossentropy as it's male/female classification (binary)
#     model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
#      # print summary of the model
#     model.summary()
#     return model
# model = create_model()

#  # use tensorboard to view metrics
# tensorboard = TensorBoard(log_dir="logs")
#  # define early stopping to stop training after 5 epochs of not improving
# early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

# batch_size = 64
# epochs = 100
#  # train the model using the training set and validating using validation set
# model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
#            callbacks=[tensorboard, early_stopping])
# model.save("results/model.h5")
#  # evaluating the model using the testing set
# print(f"Evaluating the model using {len(data['X_test'])} samples...")
# loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# print(f"Loss: {loss:.4f}")
# print(f"Accuracy: {accuracy*100:.2f}%")

# import librosa
# import numpy as np


# import argparse
from tensorflow import keras
# model = create_model()
 # load the saved/trained weights
#model.load_weights("results/model.h5")
model = keras.models.load_model("results/model.h5")

print("after loop")
print(words)
from punctuator import Punctuator
p = Punctuator('C:\Py-ML\data\Demo-Europarl-EN.pcl')
print(p.punctuate(words))
capitalwords=p.punctuate(words)
strAllTexts =capitalwords

print('Done ...')

# print file text
print('\n*** File Text ***')
# file text
print(strAllTexts)
# object type
print(type(strAllTexts))

# summary of n lines
print('\n*** n Line Summary ***')
nLineSmry = 5
print(nLineSmry)

#############################################################
# compute word freq & word weight
#############################################################

# split into words
print('\n*** Split Text To Words ***')
#import nltk
from nltk.tokenize import word_tokenize
# split 
lstAllWords = word_tokenize(strAllTexts)
# print file text
print(lstAllWords)
# print object type
print(type(lstAllWords))

# Convert the tokens into lowercase: lower_tokens
print('\n*** Convert To Lower Case ***')
lstAllWords = [t.lower() for t in lstAllWords]
print(lstAllWords)

# retain alphabetic words: alpha_only
print('\n*** Remove Punctuations & Digits ***')
import string
lstAllWords = [t.translate(str.maketrans('','','01234567890')) for t in lstAllWords]
lstAllWords = [t.translate(str.maketrans('','',string.punctuation)) for t in lstAllWords]
print(lstAllWords)

# remove all stop words
# original found at http://en.wikipedia.org/wiki/Stop_words
print('\n*** Remove Stop Words ***')
import nltk.corpus
lstStopWords = nltk.corpus.stopwords.words('english')
lstAllWords = [t for t in lstAllWords if t not in lstStopWords]
print(lstAllWords)

# remove all bad words ...
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
print('\n*** Remove Profane Words ***')
lstBadWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
lstAllWords = [t for t in lstAllWords if t not in lstBadWords]
print(lstAllWords)

# remove application specific words
print('\n*** Remove App Specific Words ***')
lstSpecWords = ['rt','via','http','https','mailto']
lstAllWords = [t for t in lstAllWords if t not in lstSpecWords]
print(lstAllWords)

# retain words with len > 3
print('\n*** Remove Short Words ***')
lstAllWords = [t for t in lstAllWords if len(t)>3]
print(lstAllWords)

# import WordNetLemmatizer
# https://en.wikipedia.org/wiki/Stemming
# https://en.wikipedia.org/wiki/Lemmatisation
# https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
print('\n*** Stemming & Lemmatization ***')
from nltk.stem import WordNetLemmatizer
# instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# Lemmatize all tokens into a new list: lemmatized
lstAllWords = [wordnet_lemmatizer.lemmatize(t) for t in lstAllWords]
print(lstAllWords)

# create a Counter with the lowercase tokens: bag of words - word freq count
print('\n*** Word Freq Count ***')
# import Counter
from collections import Counter
dctWordCount = Counter(lstAllWords)
print(dctWordCount)
print(type(dctWordCount))

# print the 10 most common tokens
print('\n*** Word Freq Count - Top 10 ***')
print(dctWordCount.most_common(10))

# word weight = word-count / max(word-count)
# replace word count with word weight
print('\n*** Word Weight ***')
max_freq = sum(dctWordCount.values())
for word in dctWordCount.keys():
    dctWordCount[word] = (dctWordCount[word]/max_freq)
# weights of words
print(dctWordCount)

#############################################################
# create sentences / lines
#############################################################

# split scene_one into sentences: sentences
print('\n*** Split Text To Sents ***')
from nltk.tokenize import sent_tokenize
lstAllSents = sent_tokenize(strAllTexts)
# print file text
print(lstAllSents)
# print object type
print(type(lstAllSents))

# print line count
print('\n*** Sents Count ***')
print(len(lstAllSents))

# convert into lowercase
print('\n*** Convert To Lower Case ***')
lstAllSents = [t.lower() for t in lstAllSents]
print(lstAllSents)

# remove punctuations
print('\n*** Remove Punctuations & Digits ***')
import string
lstAllSents = [t.translate(str.maketrans('','','[]{}<>')) for t in lstAllSents]
lstAllSents = [t.translate(str.maketrans('','','0123456789')) for t in lstAllSents]
print(lstAllSents)

# sent score
print('\n*** Sent Score ***')
dctSentScore = {}
for Sent in lstAllSents:
    for Word in nltk.word_tokenize(Sent):
        if Word in dctWordCount.keys():
            if len(Sent.split(' ')) < 30:
                if Sent not in dctSentScore.keys():
                    dctSentScore[Sent] = dctWordCount[Word]
                else:
                    dctSentScore[Sent] += dctWordCount[Word]
print(dctSentScore)


#############################################################
# summary of the article
#############################################################
# The "dctSentScore" dictionary consists of the sentences along with their scores. 
# Now, top N sentences can be used to form the summary of the article.
# Here the heapq library has been used to pick the top 5 sentences to summarize the article
print('\n*** Best Sent Score ***')
import heapq
lstBestSents = heapq.nlargest(nLineSmry, dctSentScore, key=dctSentScore.get)
for vBestSent in lstBestSents:
    print('\n'+vBestSent)
#print(type(lstBestSents))

# final summary
print('\n*** Text Summary ***')
strTextSmmry = '. '.join(lstBestSents) 
strTextSmmry = strTextSmmry.translate(str.maketrans(' ',' ','\n'))
print(strTextSmmry)
#st.write("summarization")
#st.write(strTextSmmry)
from googletrans import Translator
translator=Translator()
result = translator.translate(strTextSmmry,dest="hi")
print(result.text)
speechsumm=result.text
#st.write(result.text)
speechtrans="Speech Transaltion"

# #st.write(result.text)
if selectbox=="Speech Summarization":
    st.markdown("<h5 style='text-align:left; color:Blue; font-weight: bold;font-style: italic; font-size:25px;'>Speech", unsafe_allow_html=True)
    st.write(words)
    st.markdown("<h5 style='text-align:left; color:Blue; font-weight: bold;font-style: italic; font-size:25px;'>Speech Summarization", unsafe_allow_html=True)
    st.write(strTextSmmry)
    st.balloons()
elif selectbox=="Speech Translation":
    st.markdown("<h5 style='text-align:left; color:Blue; font-weight: bold;font-style: italic; font-size:25px;'>Speech Translation", unsafe_allow_html=True)
    image = Image.open('C:/Py-ML/data/images1.jpeg')
    st.image(image)
    st.write(result.text)
    st.balloons()
elif selectbox=="Gender Detection":
    st.balloons()
    def extract_feature(file_name, **kwargs):
          """
          Extract feature from audio file `file_name`
              Features supported:
                  - MFCC (mfcc)
                  - Chroma (chroma)
                  - MEL Spectrogram Frequency (mel)
                  - Contrast (contrast)
                  - Tonnetz (tonnetz)
              e.g:
              `features = extract_feature(path, mel=True, mfcc=True)`
          """
          mfcc = kwargs.get("mfcc")
          chroma = kwargs.get("chroma")
          mel = kwargs.get("mel")
          contrast = kwargs.get("contrast")
          tonnetz = kwargs.get("tonnetz")
          X, sample_rate = librosa.core.load(file_name)
          if chroma or contrast:
              stft = np.abs(librosa.stft(X))
          result = np.array([])
          if mfcc:
              mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
              result = np.hstack((result, mfccs))
          if chroma:
              chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
              result = np.hstack((result, chroma))
          if mel:
              mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
              result = np.hstack((result, mel))
          if contrast:
              contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
              result = np.hstack((result, contrast))
          if tonnetz:
              tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
              result = np.hstack((result, tonnetz))
          return result

      ######

    import pyaudio
    import os
    import wave
    import librosa
    import numpy as np
    from sys import byteorder
    from array import array
    from struct import pack


    THRESHOLD = 500
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RATE = 16000

    SILENCE = 30

    def is_silent(snd_data):
          "Returns 'True' if below the 'silent' threshold"
          return max(snd_data) < THRESHOLD

    def normalize(snd_data):
          "Average the volume out"
          MAXIMUM = 16384
          times = float(MAXIMUM)/max(abs(i) for i in snd_data)

          r = array('h')
          for i in snd_data:
              r.append(int(i*times))
          return r

    def trim(snd_data):
          "Trim the blank spots at the start and end"
          def _trim(snd_data):
              snd_started = False
              r = array('h')

              for i in snd_data:
                  if not snd_started and abs(i)>THRESHOLD:
                      snd_started = True
                      r.append(i)

                  elif snd_started:
                      r.append(i)
              return r

          # Trim to the left
          snd_data = _trim(snd_data)

          # Trim to the right
          snd_data.reverse()
          snd_data = _trim(snd_data)
          snd_data.reverse()
          return snd_data

    def add_silence(snd_data, seconds):
          "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
          r = array('h', [0 for i in range(int(seconds*RATE))])
          r.extend(snd_data)
          r.extend([0 for i in range(int(seconds*RATE))])
          return r

    def record():
          """
          Record a word or words from the microphone and 
          return the data as an array of signed shorts.
          Normalizes the audio, trims silence from the 
          start and end, and pads with 0.5 seconds of 
          blank sound to make sure VLC et al can play 
          it without getting chopped off.
          """
          p = pyaudio.PyAudio()
          stream = p.open(format=FORMAT, channels=1, rate=RATE,
              input=True, output=True,
              frames_per_buffer=CHUNK_SIZE)

          num_silent = 0
          snd_started = False

          r = array('h')

          while 1:
              # little endian, signed short
              snd_data = array('h', stream.read(CHUNK_SIZE))
              if byteorder == 'big':
                  snd_data.byteswap()
              r.extend(snd_data)

              silent = is_silent(snd_data)

              if silent and snd_started:
                  num_silent += 1
              elif not silent and not snd_started:
                  snd_started = True

              if snd_started and num_silent > SILENCE:
                  break

          sample_width = p.get_sample_size(FORMAT)
          stream.stop_stream()
          stream.close()
          p.terminate()

          r = normalize(r)
          r = trim(r)
          r = add_silence(r, 0.5)
          return sample_width, r

    def record_to_file(path):
          "Records from the microphone and outputs the resulting data to 'path'"
          sample_width, data = record()
          data = pack('<' + ('h'*len(data)), *data)

          wf = wave.open(path, 'wb')
          wf.setnchannels(1)
          wf.setsampwidth(sample_width)
          wf.setframerate(RATE)
          wf.writeframes(data)
          wf.close()



    def extract_feature(file_name, **kwargs):
          """
          Extract feature from audio file `file_name`
              Features supported:
                  - MFCC (mfcc)
                  - Chroma (chroma)
                  - MEL Spectrogram Frequency (mel)
                  - Contrast (contrast)
                  - Tonnetz (tonnetz)
              e.g:
              `features = extract_feature(path, mel=True, mfcc=True)`
          """
          mfcc = kwargs.get("mfcc")
          chroma = kwargs.get("chroma")
          mel = kwargs.get("mel")
          contrast = kwargs.get("contrast")
          tonnetz = kwargs.get("tonnetz")
          X, sample_rate = librosa.core.load(file_name)
          if chroma or contrast:
              stft = np.abs(librosa.stft(X))
          result = np.array([])
          if mfcc:
              mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
              result = np.hstack((result, mfccs))
          if chroma:
              chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
              result = np.hstack((result, chroma))
          if mel:
              mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
              result = np.hstack((result, mel))
          if contrast:
              contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
              result = np.hstack((result, contrast))
          if tonnetz:
              tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
              result = np.hstack((result, tonnetz))
          return result




    # load the dataset
    st.markdown("<h5 style='text-align:left; color:Blue; font-weight: bold;font-style: italic; font-size:25px;'>Gender Detection", unsafe_allow_html=True)
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities::: Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
    st.write(f"Probabilities::: Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")

    st.write("Gender Detected as:"+gender)
    if gender=="Male":
        image = Image.open('C:/Py-ML/data/boy.jpeg')
        st.image(image)
    else:
        image = Image.open('C:/Py-ML/data/girl.jpeg')
        st.image(image)
    