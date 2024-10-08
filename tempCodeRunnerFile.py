# Prediciton of readmission labels (no-readmission, 0-6 months, 6+ months):
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)

# Evaluating model
y_pred = model.predict(X_test);
score = model.evaluate(X_test, y_test,verbose=1)
roc_auc_score(y_test, y_pred);