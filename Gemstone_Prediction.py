import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU

from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt

import plotly.graph_objects as go
# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Data exploration
def explore_data(data):
    st.subheader("Data Exploration")
    st.write("Shape of the dataset:", data.shape)
    
    data.set_index('Date', inplace=True)

# Create Plotly figure
    fig = go.Figure()

# Add trace for the time series data
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Price'))

# Set layout options
    fig.update_layout(
    title='Time Series Plot of Price',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white'
    )

# Display plot using Streamlit
    st.plotly_chart(fig)

# Scale data
def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Price']])
    return scaler, scaled_data

# Create sequences
def create_sequences(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Train model
def train_model(model_type, x_train, y_train, epochs=50, batch_size=32,verbose=1):
    model = Sequential()
    if model_type == 'LSTM1':
        model.add(LSTM(128, input_shape=(1, 15)))
        model.add(Dense(1))
        optimizer1 = Adam(learning_rate=0.005)
        model.compile(loss='mean_squared_error', optimizer=optimizer1)
        model.fit(np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])), y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    elif model_type=='LSTM2':
        model.add(LSTM(128, return_sequences=True, input_shape=(1, 15)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        optimizer1 = Adam(learning_rate=0.005)
        model.compile(loss='mean_squared_error', optimizer=optimizer1)
        model.fit(np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])), y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    elif model_type == 'GRU':
        model.add(GRU(128, input_shape=(1, 15)))
        model.add(Dense(1))
        optimizer1 = Adam(learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer=optimizer1)
        model.fit(np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])), y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    elif model_type=='ANN':
        model.add( Dense(128, activation='relu', input_shape=(15,)))
        model.add(Dense(1, activation='sigmoid'))
        optimizer1 = Adam(learning_rate=0.05)
        model.compile(loss='mean_squared_error', optimizer=optimizer1)
        model.fit(x_train, y_train, epochs=75)
   
    return model

# Predict and evaluate model
def predict_evaluate_model(model, x_test, y_test, scaler):
 
    test_predict = scaler.inverse_transform(model.predict(np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))))
    test_score1=np.sqrt(mean_squared_error(scaler.inverse_transform([y_test])[0], test_predict[:,0]))
    test_score2=mean_absolute_error(scaler.inverse_transform([y_test])[0], test_predict[:,0])
    return test_predict,test_score1,test_score2

def predict_evaluate_model_ann(model, x_test, y_test, scaler):
    predict_ann = scaler.inverse_transform(model.predict(x_test))
    ann_score1= np.sqrt(mean_squared_error(scaler.inverse_transform([y_test])[0], predict_ann))
    ann_score2=mean_absolute_error(scaler.inverse_transform([y_test])[0], predict_ann)
    return predict_ann,ann_score1,ann_score2
# Plot predictions
def plot_predictions(data, predicted_data, model_name):
    col=['Price']
    Predicted_df = pd.DataFrame(predicted_data,columns=col)
    
    fig = go.Figure()

    # Add trace for actual price
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Actual Price', line=dict(color='blue')))

    # Add trace for predicted price
    fig.add_trace(go.Scatter(x=data.index, y=Predicted_df['Price'], mode='lines', name='Predicted Price', line=dict(color='red')))

    # Set layout options
    fig.update_layout(
        title=model_name,
        xaxis_title='Date',
        yaxis_title='Price of gemstone (USD/Oz)'
    )

    # Display plot using Streamlit
    st.plotly_chart(fig)
    
def forecast_prices(models, X_test, scaler, n_days):
    """
    Forecast prices for the next n_days using the given models.
    
    Args:
    - models (list): List of trained models.
    - X_test (numpy array): Test dataset for the models.
    - scaler (object): Scaler object used for normalization.
    - n_days (int): Number of days to forecast.
    
    Returns:
    - forecast_prices (dict): Dictionary containing forecasted prices for each model.
    """
    forecast_prices = {}
    for model_name, model in models.items():
        # Forecast prices for the next n_days
        forecast = []
        x_input = X_test[-1]  # Get the last sequence in the test data
        
        for i in range(n_days):
            # Reshape the input sequence for prediction
            x_input_reshaped = x_input.reshape((1, 1, x_input.shape[0]))
            
            # Predict the next price
            y_hat = model.predict(x_input_reshaped, verbose=0)
            
            # Inverse transform the predicted price
            y_hat_inversed = scaler.inverse_transform([[y_hat[0][0]]])[0]
            
            # Append the predicted price to the forecast list
            forecast.append(y_hat_inversed)
            
            # Update the input sequence for the next prediction
            x_input = np.append(x_input[1:], y_hat)
            
        # Store the forecasted prices for the model
        forecast_prices[model_name] = forecast
    
    return forecast_prices

def predict_prices(model,X_test, scaler, n_days):
    forecast = []
    x_input = X_test[-1]  # Get the last sequence in the test data
    
    for i in range(n_days):
        # Reshape the input sequence for prediction
        x_input_reshaped = x_input.reshape((1, 1, x_input.shape[0]))
        
        # Predict the next price
        y_hat = model.predict(x_input_reshaped, verbose=0)
        
        # Inverse transform the predicted price
        y_hat_inversed = scaler.inverse_transform([[y_hat[0][0]]])[0]
        
        # Append the predicted price to the forecast list
        forecast.append(y_hat_inversed)
        
        # Update the input sequence for the next prediction
        x_input = np.append(x_input[1:], y_hat)
    return forecast


@st.cache_data
def calculated_forecasted_price(_n_days,_x_test,_lstm_model1,_lstm_model2,_gru_model,_scaler):
    lstm_forecast1 = forecast_prices(models={'LSTM 1': _lstm_model1}, X_test=_x_test, scaler=_scaler, n_days=_n_days)
    lstm_forecast2 = forecast_prices(models={'LSTM 2': _lstm_model2}, X_test=_x_test, scaler=_scaler, n_days=_n_days)
    gru_forecast = forecast_prices(models={'GRU': _gru_model}, X_test=_x_test, scaler=_scaler, n_days=_n_days)
    return lstm_forecast1,lstm_forecast2,gru_forecast

def plot_predictions_with_forecast(data, lstm_forecast1,lstm_forecast2,gru_forecast):
    col=['Price']
    
    
    fig = go.Figure()

    # Add trace for actual price
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Actual Price'))

   
    # Add traces for forecasted prices of each model
    for model, prices in lstm_forecast1.items():
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(prices), freq='D')
        # Plot the forecasted prices
        
        fig.add_trace(go.Scatter(x=forecast_dates, y=prices, mode='lines', name=f'{model} Forecast',line=dict(color='blue')))
    for model, prices in lstm_forecast2.items():
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(prices), freq='D')
        # Plot the forecasted prices
        
        fig.add_trace(go.Scatter(x=forecast_dates, y=prices, mode='lines', name=f'{model} Forecast',line=dict(color='red')))
    for model, prices in gru_forecast.items():
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(prices), freq='D')
        # Plot the forecasted prices
        
        fig.add_trace(go.Scatter(x=forecast_dates, y=prices, mode='lines', name=f'{model} Forecast',line=dict(color='green')))

    # Set layout options
    fig.update_layout(
        title="Forecasted Prices",
        xaxis_title='Date',
        yaxis_title='Price of gemstone'
    )

    # Display plot using Streamlit
    st.plotly_chart(fig)

def add_forecast_to_data( data,forecasted_prices):
    # Get the last date in the existing data
    last_date = data.index[-1]
    
    # Generate forecast dates starting from the day after the last date
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecasted_prices), freq='D')
    
    # Create a DataFrame for forecasted data
    forecast_data = pd.DataFrame({'Date': forecast_dates, 'Price': forecasted_prices})
    forecast_data['Price'] = forecast_data['Price'].astype(float)
    # Set the 'Date' column as the index
    forecast_data.set_index('Date', inplace=True)
    

    
    
    
    return forecast_data


def plot_predictions_with_forecast_2(data,  lstm1_forecasted_prices,lstm2_forecasted_prices,gru_forecasted_prices):
    lstm1forecasteddata = add_forecast_to_data(data, lstm1_forecasted_prices)
    lstm2forecasteddata = add_forecast_to_data(data, lstm2_forecasted_prices)
    gruforecasteddata = add_forecast_to_data(data, gru_forecasted_prices)
    #data['Forecasted_Prices'] = forecasted_prices
    

    
    
    # Add trace for actual price
    actual_trace = go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Actual Price', line=dict(color='blue'))
    forecast_trace1 = go.Scatter(x=lstm1forecasteddata.index, y=lstm1forecasteddata['Price'], mode='lines', name='LSTM 1 Forecast Price', line=dict(color='orange'))
    forecast_trace2 = go.Scatter(x=lstm2forecasteddata.index, y=lstm2forecasteddata['Price'], mode='lines', name='LSTM 2 Forecast Price', line=dict(color='yellow'))
    forecast_trace3= go.Scatter(x=gruforecasteddata.index, y=gruforecasteddata['Price'], mode='lines', name='GRU Forecast Price', line=dict(color='green'))

    # Add traces for forecasted prices of each model
    
    # Plot the forecasted prices
    
   
    fig = go.Figure()

    # Add actual and forecasted traces to the figure
    fig.add_trace(actual_trace)
    fig.add_trace(forecast_trace1)
    fig.add_trace(forecast_trace2)
    fig.add_trace(forecast_trace3)
    fig.add_shape(type="line",
                  x0=data.index[-1], y0=min(data['Price']),
                  x1=lstm1forecasteddata.index[0], y1=min(lstm1forecasteddata['Price']),
                  line=dict(color="cyan", width=1, dash="dash"),
                  )
    fig.add_shape(type="line",
                  x0=data.index[-1], y0=max(data['Price']),
                  x1=lstm1forecasteddata.index[0], y1=max(lstm1forecasteddata['Price']),
                  line=dict(color="cyan", width=1, dash="dash"),
                  )
    # Set layout options
    fig.update_layout(
        title="Forecasted Prices",
        xaxis_title='Date',
        yaxis_title='Price of gemstone'
    )

    # Display plot using Streamlit
    st.plotly_chart(fig)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def main():
    st.title("Gemstone Price Prediction")

    # Gemstone selection
    gemstone_options = ['Gold', 'Platinum', 'Silver']
    selected_gemstone = st.selectbox('Select Gemstone', gemstone_options)

    # Load data based on selected gemstone
    if selected_gemstone == 'Gold':
        data_file_path = 'Gold2.xlsx'
    elif selected_gemstone == 'Platinum':
        data_file_path = 'Platinum2.xlsx'
    elif selected_gemstone == 'Silver':
        data_file_path = 'Silver2.xlsx'
    else:
        st.error('Invalid gemstone selection!')
        return

    data = load_data(data_file_path)

    # Data exploration
    explore_data(data)

    # Scale data
    scaler, scaled_data = scale_data(data)

    # Split data into train and test sets
    training_size = round(len(data) * 0.80)
    train_data = scaled_data[:training_size]
    test_data = scaled_data[training_size:]

    # Create sequences
    x_train, y_train = create_sequences(train_data)
    x_test, y_test = create_sequences(test_data)

    df1 = data[training_size + 15+ 1:]
    #regressor = KerasRegressor(build_fn=create_lstm_model, epochs=50, batch_size=32, verbose=0)
    #param_grid = {
    #    'dropout_rate': [0.0,0.1, 0.2, 0.3],
    #    'learning_rate': [0.001,0.005,0.01,0.05, 0.1],
    #    'optimizer': ['adam', 'rmsprop'],
    #    'look_back': [7, 15, 30],
     #   'batch_size': [32, 64,128],
      #  'epochs': [50,75,100,150],
      #  'activation': ['relu', 'tanh']
    #}
    #grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3)
    #grid_result = grid_search.fit(x_train, y_train)

# Summarize results
    #best_score = grid_result.best_score_
    #best_params = grid_result.best_params_
    #st.write("Best score:", best_score)
    #st.write("Best parameters:", best_params)

    # Train and evaluate models
    #lstm_model1 = train_model('LSTM1', x_train, y_train)
    #lstm_model2 = train_model('LSTM2', x_train, y_train, epochs=100)
    #gru_model = train_model('GRU', x_train, y_train)
    #ann_model = train_model('ANN', x_train, y_train)
    #lstm_model1.save('./'+selected_gemstone+'lstm_model1.h5')
    #lstm_model2.save('./'+selected_gemstone+'lstm_model2.h5')
    #gru_model.save('./'+selected_gemstone+'gru_model.h5')
    #ann_model.save('./'+selected_gemstone+'ann_model.h5')
    models = {
        'LSTM 1': 'lstm_model1_final',
        'LSTM 2': 'lstm_model2_final',
        'GRU': 'gru_model_final',
        'ANN':'ann_model_final'    
    }
    evaluation_results=[]
    for model_name, model in models.items():
    
        st.subheader(f"{model_name} Model")
        model_pth = './'+selected_gemstone+models[model_name]+'.h5'
        model = tf.keras.models.load_model(model_pth)
        if model_name=="ANN":
            predicted,score1a, score1b=predict_evaluate_model_ann(model, x_test, y_test, scaler)
            
            plot_predictions(df1, predicted, f'{model_name} Model')
        else:
            predicted,score1a, score1b= predict_evaluate_model(model, x_test, y_test, scaler)
            
            plot_predictions(df1, predicted, f'{model_name} Model')
        evaluation_results.append({
            'Model': model_name,
            'RMSE': score1a,
            'MAE': score1b
        })

        
        

    # Create DataFrame for evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)

    # Display evaluation results in a table
    st.subheader("Evaluation Results of : "+selected_gemstone)
    st.write(evaluation_df)

    
    lstm_model1 = tf.keras.models.load_model('./'+selected_gemstone+'lstm_model1_final'+'.h5')
    lstm_model2=tf.keras.models.load_model('./'+selected_gemstone+'lstm_model2_final'+'.h5')
    gru_model=tf.keras.models.load_model('./'+selected_gemstone+'gru_model_final'+'.h5')

#forecasting prices for each model and each day up to 3mnths/90days
    n_days = st.number_input('Enter the number of days to forecast', min_value=1, max_value=90, value=14)
    #lstm_forecast1,lstm_forecast2,gru_forecast=calculated_forecasted_price(n_days,x_test,lstm_model1,lstm_model2,gru_model,scaler)
    lstm1_forecasted_prices = predict_prices(lstm_model1,x_test,scaler,n_days)
    lstm2_forecasted_prices = predict_prices(lstm_model2,x_test,scaler,n_days)
    gru_forecasted_prices = predict_prices(gru_model,x_test,scaler,n_days)
    st.subheader("Forecasted Prices of : "+selected_gemstone)


    # Display forecasted prices
    #plot_predictions_with_forecast(data, lstm_forecast1,lstm_forecast2,gru_forecast)
    plot_predictions_with_forecast_2(df1,lstm1_forecasted_prices,lstm2_forecasted_prices,gru_forecasted_prices)
    
    
    #for model_name, prices in lstm_forecast1.items():
    #    st.write(f"{model_name} Forecast:", prices)
    #for model_name, prices in lstm_forecast2.items():
    #    st.write(f"{model_name} Forecast:", prices)
    #for model_name, prices in gru_forecast.items():
    #    st.write(f"{model_name} Forecast:", prices)
    
if __name__ == "__main__":
    main()


