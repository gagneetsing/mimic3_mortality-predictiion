"""
MIMIC-III Project
@author: Daniel Solá
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath('__file__')));

import pandas as pd
from features.get_features import *
from labels.get_labels import *
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, accuracy_score
from services.plotting_service import *
# from tensorflow.python.trackable import base as trackable
from sklearn.preprocessing import StandardScaler


# Extraction of features
physioa = Features().Measures().get_measures()
demographic = Features().DemographicData().get_demographic_data()
icu = Features().ICUData().get_icu_data()

features = [physioa, demographic, icu]
physio = reduce(
    lambda left, right: pd.merge(left, right, on="hadm_id", how="outer"), features
)
print("Printing physio")
print(physio)
variable = physio.avg_spo2
# PlottingService().plot_kde(variable, 'Temperature', 'ºF', 'Relative Frequency (%)',1000)
categorical_labels = PatientOutcomes().get_categorical_outcomes();
label = pd.DataFrame(categorical_labels.mortality);
print(f"Label\n{label}")
physio = pd.DataFrame(physio)
print("Printing physio")
print(physio.head(5))
# data = pd.merge(physio, label, left_index = True, right_index = True, how = 'inner').dropna();
data = pd.concat([physio, label])
print("Dataframe")
print(data.head(5))
print(data.shape)
#data.drop(columns={'gender','marital_status','religion','ethnicity','service','icd9_group'},inplace=True)
# Spliting of data in test / train sets
features = data.loc[:, "hadm_id":"procedure_count"]
print(features.info())

features = pd.get_dummies(features)
print(features.info())
label = data.mortality;
hot_encoded_label = pd.get_dummies(label);

print("Features shape:", features.shape)
print("Hot encoded label shape:", hot_encoded_label)


X_train, X_test, y_train, y_test = train_test_split(features, hot_encoded_label, test_size=0.075, random_state=42)


print(f"{X_train.shape[1]}")

# Prediciton of readmission labels (no-readmission, 0-6 months, 6+ months):
# model = Sequential()
# model.add(Dense(64, activation="relu", input_shape=(13,)))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(3, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1, validation_split=0.1)

from keras.initializers import he_normal
from keras.optimizers import AdamW,SGD

# Neural Network Architecture with He initialization
model = Sequential()
model.add(
    Dense(
        128,
        activation="relu",
        kernel_initializer=he_normal(),
        input_shape=(X_train.shape[1],),
    )
)

model.add(Dropout(0.3))
model.add(Dense(64, activation="relu", kernel_initializer=he_normal()))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu", kernel_initializer=he_normal()))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu", kernel_initializer=he_normal()))
model.add(Dense(3, activation="softmax"))

# Reduce learning rate
optimizer = AdamW(learning_rate=0.0001)

# Compile the model
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

# Train the model with a smaller batch size
model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.1)

# Evaluating model
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {score[0]}, Test Accuracy: {score[1]}")

# Evaluate using ROC AUC score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred, multi_class="ovr"))


# # Evaluating model
# y_pred = model.predict(X_test);
# score = model.evaluate(X_test, y_test,verbose=1)
# print(roc_auc_score(y_test, y_pred));

# # from sklearn.metrics import roc_auc_score

# # roc = roc_auc_score(y_test, y_pred)
# # print("ROC_AUC_SCORE:", roc)
# # ROC_AUC_SCORE = 0.866

# Neural Network Architecture
# model = Sequential()
# model.add(
#     Dense(64, activation="relu", input_shape=(13,))
# )  # Increased neurons in the input layer
# # model.add(Dropout(0.3))  # Add dropout to reduce overfitting
# model.add(Dense(64, activation="relu"))
# # model.add(Dropout(0.3))
# model.add(Dense(32, activation="relu"))
# # model.add(Dropout(0.2))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(16, activation="relu"))
# model.add(
#     Dense(3, activation="softmax")
# )  # Using softmax for multi-class classification

# # Compile the model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Train the model
# model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.1)

# # Evaluating model on the test data
# y_pred = model.predict(X_test)
# score = model.evaluate(X_test, y_test, verbose=1)
# print(f"Test Loss: {score[0]}, Test Accuracy: {score[1]}")

# # Evaluate using ROC AUC score
# print("ROC AUC Score:", roc_auc_score(y_test, y_pred, multi_class="ovr"))
