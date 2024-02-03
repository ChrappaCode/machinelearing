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


#ZDROJE: Kódy zo seminárov 2,3 od pani cvičiacej
#ZDROJE: Hlavne EDA s pomocou ChatGPT

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie1_dataset.csv')
df_eda = pd.read_csv('zadanie1_dataset.csv')
df_last = pd.read_csv('zadanie1_dataset.csv')

print("*"*100, "Before removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

df = df[df['duration_ms'] > 0]
df_eda = df[df['duration_ms'] > 0]
df_last = df[df['duration_ms'] > 0]
df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]
df_eda = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]
df_last = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]

print("*"*100, "After removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

print("*"*100, "Missing values", "*"*100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

df.dropna(inplace=True)
df_eda.dropna(inplace=True)
df_last.dropna(inplace=True)

print("*"*100, "Missing values after removing them", "*"*100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

le = LabelEncoder()
df['explicit'] = le.fit_transform(df['explicit'])
df['emotion'] = le.fit_transform(df['emotion'])

le2 = LabelEncoder()
df_last['explicit'] = le2.fit_transform(df_last['explicit'])

print("*"*100, "Label encoding", "*"*100)
print(df['explicit'].head(10))
print("*"*100, "Label encoding", "*"*100)
print(df['emotion'].head(20))

dummies = pd.get_dummies(df['top_genre'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('top_genre', axis=1), dummies], axis=1)

print("*" * 100, "Dummy encoding", "*" * 100)
print(df.head(10))

#------------------------------------------------
dummies = pd.get_dummies(df_last['top_genre'])
dummies = dummies.astype(int)
df_last = pd.concat([df_last.drop('top_genre', axis=1), dummies], axis=1)
#------------------------------------------------------------------

df.drop(columns=['url'], inplace=True)
df.drop(columns=['name'], inplace=True)
df.drop(columns=['filtered_genres'], inplace=True)
df.drop(columns=['genres'], inplace=True)


df_last.drop(columns=['url'], inplace=True)
df_last.drop(columns=['name'], inplace=True)
df_last.drop(columns=['filtered_genres'], inplace=True)
df_last.drop(columns=['genres'], inplace=True)
#df.drop(columns=['top_genre'], inplace=True) <––––––


print("*"*100, "Column types", "*"*100)
print(df.dtypes)

# Split dataset into X and y
X = df.drop(columns=['emotion'])
y = df['emotion']

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# Print dataset shapes
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# Plot histograms before scaling#
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*100, "Before scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Scale data
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()

print("*"*100, "After scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Train MLP model to predict country
print("*"*100, "MLP", "*"*100)
print(f"Random accuracy: {1/len(y_train.unique())}")

clf = MLPClassifier(
    hidden_layer_sizes=(100, 100, 5, 6, 90),
    random_state=1,
    max_iter=10,
    validation_fraction=0.2,
    early_stopping=True,
    learning_rate='adaptive',
    learning_rate_init=0.001,
).fit(X_train, y_train)

# Predict on train set
y_pred = clf.predict(X_train)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
cm_train = confusion_matrix(y_train, y_pred)

# Predict on test set
y_pred = clf.predict(X_test)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
cm_test = confusion_matrix(y_test, y_pred)

# Create class names for confusion matrix
class_names = list(le.inverse_transform(clf.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

sns.set(style="whitegrid")

correlation_matrix = df.corr()
plt.figure(figsize=(25, 20))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()



plt.figure(figsize=(10, 6))
sns.violinplot(x="emotion", y="danceability", data=df_eda, inner="stick", color="yellow")
plt.title("Violin Plot of Danceability by Emotion")
plt.show()


print(df_eda["energy"])
plt.figure(figsize=(10, 6))
sns.swarmplot(x="energy", y="loudness", data=df_eda, palette="husl")
plt.title("Swarm Plot of Loudness by Energy")
custom_ticks = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 1900]
custom_ticks2 = [0, 0.125, 0.250, 0.375, 0.5, 0.625, 0.750, 0.875, 1]
plt.xticks(custom_ticks, custom_ticks2)
plt.show()


text_data = df_eda['name'].str.cat(sep=' ')
wordcloud = WordCloud(width=700, height=400, background_color='white').generate(text_data)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most used words in song names')
plt.show()

fig = px.histogram(df_eda, x='liveness', color='top_genre', marginal='box')
fig.update_layout(title='Interactive Histogram with Box Plot showing liveness to genre')
fig.show()


fig = px.scatter(df_eda, x='valence', y='number_of_artists', color='acousticness',
                 size='tempo', hover_data=['name', 'url'])
fig.update_layout(title='Interactive Scatter Plot showing valence to number of artists and coloring to acousticness on hover showing name and URL')
fig.show()

plt.figure(figsize=(10, 6))
plt.plot(df_eda['emotion'], df_eda['explicit'], label='Sine Wave', color='b', linestyle='--', marker='o', markersize=8)
plt.title('Fancy EDA Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()



labels = df_eda['emotion'].unique()  # Get unique values from the column
sizes = df_eda['emotion'].value_counts()  # Count occurrences of each unique value

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Pie Chart of Emotions')
plt.axis('equal')

plt.show()


# Časť 3

X = df_last.drop(columns=['emotion'])
y = df_last['emotion']
y = pd.get_dummies(y)
y = y.astype(int)

print(y)

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

model = Sequential()

model.add(Dense(24, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=500, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
test_scores = model.evaluate(X_test, y_test, verbose=0)

print("*"*100, "Test accuracy", "*"*100)
print(f"Test accuracy: {test_scores[1]:.4f}")

train_scores = model.evaluate(X_train, y_train, verbose=0)

print("*"*100, "Train accuracy", "*"*100)
print(f"Train accuracy: {train_scores[1]:.4f}")

# Plot confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5)


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

class_names = df_last['emotion'].unique().tolist()

cm = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax_ = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax_)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()


# Plot loss and accuracy
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Plot loss and accuracy")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Plot loss and accuracy")
plt.legend()
plt.show()
