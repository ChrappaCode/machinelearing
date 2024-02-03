import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import keras
from keras.src.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import hamming_loss, f1_score
from wordcloud import WordCloud

df = pd.read_csv('zadanie1_dataset.csv')

df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]
df = df[(df['energy'] >= 0) & (df['energy'] <= 1)]
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]
df = df[(df['speechiness'] >= 0) & (df['speechiness'] <= 1)]
df = df[(df['acousticness'] >= 0) & (df['acousticness'] <= 1)]
df = df[(df['duration_ms'] >= 0) & (df['duration_ms'] <= 2000000)]
df = df[(df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1)]
df = df[(df['liveness'] >= 0) & (df['liveness'] <= 1)]
df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]
df = df[(df['popularity'] >= 0) & (df['popularity'] <= 100)]

df = df.drop(columns=['explicit'])
df = df.drop(columns=['name'])
df = df.drop(columns=['url'])
df = df.drop(columns=['genres'])
df = df.drop(columns=['filtered_genres'])
df = df.drop(columns=['top_genre'])

df = df.drop(columns=['number_of_artists'])
df = df.drop(columns=['popularity'])
df = df.drop(columns=['duration_ms'])

df.dropna(inplace=True)

X = df.drop(columns=['emotion'])
y = pd.get_dummies(df['emotion'])
y = y.astype(int)
# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

model = Sequential()
model.add(Dense(24, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# # Define EarlyStopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=700, batch_size=32)
                    # , callbacks=[early_stopping])

test_scores = model.evaluate(X_test, y_test, verbose=0)
train_scores = model.evaluate(X_train, y_train, verbose=0)

print("*"*100, "Test accuracy", "*"*100)
print(f"Test accuracy: {test_scores[1]:.4f}")
print(f"Train accuracy: {train_scores[1]:.4f}")

y_pred_test = model.predict(X_test)
y_pred_test = np.argmax(y_pred_test, axis=1)

class_names = df['emotion'].unique().tolist()

cm_test = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp_test.plot(ax=ax)
disp_test.ax_.set_title("Confusion matrix on test set")
disp_test.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()
# Predictions on the training set
y_pred_train = model.predict(X_train)
y_pred_train = np.argmax(y_pred_train, axis=1)

cm_train = confusion_matrix(np.argmax(y_train.values, axis=1), y_pred_train)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp_train.plot(ax=ax)
disp_train.ax_.set_title("Confusion matrix on train set")
disp_train.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()