import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def arx_model(df,
              model,
              number_epochs=1000, 
              p_feats=(2,2),
              p_res=2,
              features=['ppt','tmean'],
              response = 'NDVI',
              test_size = 0.1,
              val_size = 0.1,
              shuffle_test = False,
              shuffle_validation = True,
              loss = 'mean_squared_error', 
              metrics=['mean_squared_error'],
              optimizer='adam',
              callbacks =[EarlyStopping(patience=30)],
              ):
    
    # the following function is going to produce the data for feeding the model 
    def data_gen(df):
        # the labels are where the response is not none, also we took them from p_res because we do not want to 
        # predict before that  
        labels = df[response].dropna()[p_res:]
        # this are the past response values we want to use for prediction 
        past_labels = df[response].dropna()
        # using sliding_window_view function to build get a vecotr of past labels with p_res lenght
        # the result of this function is an array with length past_labels and p_res but we drop the last element 
        # as the last element wont be used for prediction 
        past_labels = sliding_window_view(past_labels,p_res)[:-1]
        # here we build the past predictors array. for each index at which we have a response 
        # we get the past values of each features with its asscociated time_window in p_feats and 
        # make an array with shape length of labels and time_window. we do this for each feature and then stack 
        # them horizontly to have an array of the shape number of labels and sum of time_windows which is sum(p_feats) 
        past_predictors = np.hstack([np.array([df[feature].iloc[index-time_window+1:index+1] 
                                               for index in labels.index]) 
                                               for feature,time_window in zip(features,p_feats)])
        # we stack past labels with past predictors to have the X vector which is all the past predictors together
        X = np.hstack((past_labels,past_predictors))
        # y is simply the labels
        y = np.array(labels)
        return X,y

    # we generate the data from the above function.
    # again: X shape is (len(labels), sum(p_feats)) and y shape is of course (len(labels),)
    X,y = data_gen(df=df)


    
    # Then split data to test, train and validation. The option shuffle is usually none here, 
    # but we can also shuffle the data before choosing test data. the problem is that we loose the time structure 
    # in the test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=shuffle_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, shuffle=shuffle_validation, random_state=42)
  
    
    # the model is compiled here, It can be linear or non-linear. the implementation is in keras.
    # we can use differernt loss functions and metrics and optimizers here.  
    model.compile(loss = loss, metrics=metrics,optimizer=optimizer)
    # here we fit model 
    history=model.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val,y_val),batch_size=32,verbose=0,callbacks=callbacks)
   
    #evualation
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    model_prediction = model.predict(X_test,verbose=0)
    test_loss = mean_squared_error(y_test,model_prediction)
    r2 = r2_score(y_test,model_prediction)


    # accuracies printing
    print(f"train loss : {train_loss[-1]:.{5}f}.")
    print(f"validation loss : {val_loss[-1]:.{5}f}.")
    print(f"test loss : {test_loss:.{5}f}.")
    print(f"r2 : {r2:.{5}f}.") 

    
    return model, history, X_test, y_test, model_prediction, r2,train_loss, test_loss