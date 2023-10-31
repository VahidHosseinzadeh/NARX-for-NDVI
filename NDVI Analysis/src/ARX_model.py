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
              fill_missing = False,
              early_stopping= EarlyStopping(patience=30),
              test_time = None):

    def features_sw(df, features,p_feats): 

        max_p = max(*p_feats,p_res+1)
        drops_feats = [max_p - p for p in p_feats] 
        # drop_res = max_p - p_res
        # sws = [sliding_window_view(df[res].shift(1).values[drop_res:],p_res)]
        feature_sws = [sliding_window_view(df[f].values[d:],p) for f,p,d in zip(features,p_feats,drops_feats)]
        # sws.extend(feature_sws)
        X= np.hstack(feature_sws)
        # y = df[res].values[max_p-1:]
        # assert len(y) == len(X)
        return X 
    
    # def sw_res_feats(df,res = 'NDVI', features=['ppt','tmean'],p_res=1,p_feats=(2,2)):
    #     res_values = df[res].values
    #     features_values = [df[f].values for f in features]
    #     v = []
    #     for index,res in enumerate(res_values):
    #             if ~np.isnan(res):
    #                 ll = [l[index-p+1:index+1] for l,p in zip(features_values,p_feats)]
    #                 v.append((index,res,ll))

    #     res = [i[1]for i in v]
    #     res_sw = sliding_window_view([i[1]for i in v],p_res)[:-1]
    #     features_sw = [np.array([v[i][2][j] for i in range(p_res,len(v))]) for j in range(len(features))]
    #     X = np.hstack([res_sw,*features_sw])
    #     y = np.array(res)[p_res:]
    #     assert len(X) == len(y)
    #     return X,y

    def data_gen(df,res = 'NDVI', features=['ppt','tmean'],p_res=1,p_feats=(1,2)):
        # the labels are where the response is not none, also we took them from p_res because we do not want to 
        # predict before that  
        labes = df[res].dropna()[p_res:]
        # this are the past response values we want to use for prediction 
        past_labels = df[res].dropna()
        # using sliding_window_view function to build get a vecotr of past labels with p_res lenght
        # the result of this function is an array with length past_labels and p_res but we drop the last element 
        # as the last element wont be used for prediction 
        past_labels = sliding_window_view(past_labels,p_res)[:-1]
        # here we build the past predictors array. for each index at which we have a response 
        # we get the past values of each features with its asscociated time_window in p_feats and 
        # make an array with shape length of labels and time_window. we do this for each feature and then stack 
        # them horizontly to have an array of the shape number of labels and sum of time_windows which is sum(p_feats) 
        past_predictors = np.hstack([np.array([df[feature].iloc[index-time_window+1:index+1] 
                                               for index in labes.index]) 
                                               for feature,time_window in zip(features,p_feats)])
        # we stack past labels with past predictors to have the X vector which is all the past predictors together
        X = np.hstack((past_labels,past_predictors))
        # y is simply the labels
        y = np.array(labes)
        return X,y

    # X,y = features_sw(df=df,features=features,res=response,p_feats=p_feats,p_res=p_res)
    # X,y = sw_res_feats(df=df,res = response, features=features,p_res=p_res,p_feats=p_feats)
    X,y = data_gen(df=df,res = response, features=features,p_res=p_res,p_feats=p_feats)
    

    if test_time != None:
        dfna = df.dropna()
        number_of_test = (dfna.reset_index()['Timestamp'] >= test_time).value_counts()[True]
        test_fraction = number_of_test/len(dfna)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction,shuffle=False)
    elif test_time == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=shuffle_test)

    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, random_state=42)
  
    

    model.compile(loss = 'mean_squared_error', metrics=['mean_squared_error'],optimizer='adam')
    history=model.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val,y_val),batch_size=32,verbose=0,callbacks=[early_stopping])
   
    #evualation
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    model_prediction = model.predict(X_test,verbose=0)
    test_loss = mean_squared_error(y_test,model_prediction)
    r2 = r2_score(y_test,model_prediction)
    
    preds_ndvi_all = None
    if fill_missing:
        preds_ndvi_all = model.predict(features_sw(df,features,p_feats),verbose=0)
        preds_ndvi_all = preds_ndvi_all.flatten()
        preds_ndvi_all = np.insert(preds_ndvi_all,0,[np.nan] * (max(p_feats)-1))


    # accuracy is 5 here
    print(f"train loss : {train_loss[-1]:.{5}f}.")
    print(f"validation loss : {val_loss[-1]:.{5}f}.")
    print(f"test loss : {test_loss:.{5}f}.")
    print(f"r2 : {r2:.{5}f}.") 

    

    long_term_forecast = None
    long_term_loss = None
    long_term_r2 = None
    if p_res > 0:
        first_window = X_test[0][:p_res]
        input_dimension = p_res+ sum(p_feats)
        X_features_test = np.hsplit(X_test,(p_res,input_dimension))[1]

        ndvis = first_window
        for i in range(len(y_test)):
            x = np.hstack((ndvis[-p_res:],X_features_test[i])).reshape(1,input_dimension)
            pred = model.predict(x,verbose=0)[0][0]
            ndvis = np.append(ndvis,pred)
        long_term_forecast = ndvis[p_res:]
        long_term_loss = mean_squared_error(y_test,long_term_forecast)
        long_term_r2 = r2_score(y_test,long_term_forecast)

        print(f"long term test loss: {long_term_loss:.{5}f}.")
        print(f"long term r2: {long_term_r2:.{5}f}.")
    
    
    return model, history, X_test, y_test, model_prediction, preds_ndvi_all, long_term_forecast, r2,train_loss, test_loss,  long_term_loss, long_term_r2