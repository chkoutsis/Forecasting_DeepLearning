import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Data Analysis
df = pd.read_csv("Electric_Production.csv")
df = df.set_index("DATE")
df.index = pd.to_datetime(df.index, format='%d-%m-%Y')
# df = df.groupby(pd.Grouper(freq='m')).mean()


print("Number of measurements: "+str(len(df['Value'])))
df.plot(y='Value', rot=25);


# Spliting the last 2 years for test
train = dftm[:12*27].values
plt.plot(np.arange(len(train)),train)
train = train.reshape((len(train), 1))
test = dftm[12*27:].values
plt.plot(np.arange(len(train), len(train)+len(test)),test)
test = test.reshape((len(test), 1))


length = 12
generator = TimeseriesGenerator(train,train,length=length, batch_size=1)
validation_generator = TimeseriesGenerator(test,test,length=length, batch_size=1)

print(train[:length+1])


# Looking some TimeSeriesGenerator results
i=0
for x,y in generator:
    print(x)
    print(y)
    i = i + 1
    if i == 2:
        break


#RNN Model
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()



epochs = 100
early_stop = EarlyStopping(monitor='val_loss',patience=10)
ckpt = ModelCheckpoint('model6.hdf5', save_best_only=True, monitor='val_loss', verbose=1)

history = model.fit_generator(
    generator,
    steps_per_epoch=len(generator),
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, ckpt])



history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(5,5))
#plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# Load the best model
model.load_weights("model6.hdf5")

# Predicting some days ahead.
test_predictions = []
first_eval_batch = train[-length:]
current_batch = first_eval_batch.reshape((1, length, 1))
for i in range(len(test)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    # store prediction
    test_predictions.append(current_pred)
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#prediction = scaler.inverse_transform(test_predictions)


# Comparing test data and predictions

plt.plot(np.arange(len(train)), train)
plt.plot(np.arange(len(train),len(train)+len(test)),test)
plt.plot(np.arange(len(train),len(train)+len(test)),test_predictions)


# Calculating the mean squared error
loss = np.mean(np.square(test[:,0] - np.array(test_predictions)[:,0]), axis=-1)
print("mse: "+str(loss))



#LSTM Model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()



epochs = 100
early_stop = EarlyStopping(monitor='val_loss',patience=10)
ckpt = ModelCheckpoint('model7.hdf5', save_best_only=True, monitor='val_loss', verbose=1)

history = model.fit_generator(
    generator,
    steps_per_epoch=len(generator),
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, ckpt])


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(5,5))
#plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# Load the best model
model.load_weights("model7.hdf5")

# Predicting some days ahead.
test_predictions = []
first_eval_batch = train[-length:]
current_batch = first_eval_batch.reshape((1, length, 1))
for i in range(len(test)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    # store prediction
    test_predictions.append(current_pred)
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#prediction = scaler.inverse_transform(test_predictions)


# Comparing test data and predictions

plt.plot(np.arange(len(train)), train)
plt.plot(np.arange(len(train),len(train)+len(test)),test)
plt.plot(np.arange(len(train),len(train)+len(test)),test_predictions)

# Calculating the mean squared error
loss2 = np.mean(np.square(test[:,0] - np.array(test_predictions)[:,0]), axis=-1)
print("mse: "+str(loss2))


#GRU Model
model = Sequential()
model.add(GRU(20, activation='relu', return_sequences=True, input_shape=(length,1)))
model.add(GRU(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()



epochs = 100
early_stop = EarlyStopping(monitor='val_loss',patience=10)
ckpt = ModelCheckpoint('model8.hdf5', save_best_only=True, monitor='val_loss', verbose=1)

history = model.fit_generator(
    generator,
    steps_per_epoch=len(generator),
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, ckpt])


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(5,5))
#plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# Load the best model
model.load_weights("model8.hdf5")

# Predicting some days ahead.
test_predictions = []
first_eval_batch = train[-length:]
current_batch = first_eval_batch.reshape((1, length, 1))
for i in range(len(test)):
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    # store prediction
    test_predictions.append(current_pred)
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#prediction = scaler.inverse_transform(test_predictions)



# Comparing test data and predictions
plt.plot(np.arange(len(train)), train)
plt.plot(np.arange(len(train),len(train)+len(test)),test)
plt.plot(np.arange(len(train),len(train)+len(test)),test_predictions)



# Calculating the mean squared error
loss3 = np.mean(np.square(test[:,0] - np.array(test_predictions)[:,0]), axis=-1)
print("mse: "+str(loss3))



Summary = {'Method':['RNN','LSTM','GRU'],
              'Mean Squared Error':[str(loss),str(loss2),str(loss3)]}

print(Summary)
