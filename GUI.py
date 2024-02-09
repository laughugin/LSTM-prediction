import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from HistoricalData import HistoricalData
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the LSTM model
model = load_model('LSTM_model.h5')
global df  # Declare df as a global variable

def load_data():
    global df  # Use the global df variable
    start_date_str = start_date_entry.get()
    end_date_str = end_date_entry.get()
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    instance = HistoricalData("BTC", "USDT", start_time=start_date, end_time=end_date)
    instance.get_data()
    
    df = pd.read_csv("BTCUSDT.csv")
    
    plot_data(df)

def plot_data(df):
    for widget in graph_frame.winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['timestamp'], df['close_price'], color='blue', linewidth=2)
    ax.set_title('BTC Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    canvas.draw()

def predict_data():
    global df
    data = df['close_price'].values.reshape(-1, 1)
    
    last_points = data[-300:]
    data_for_model = data[:-300]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_model)
    
    sequence_length = 10000  # Adjust based on your LSTM model's input shape
    current_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    
    lstm_predicted_prices = []
    for _ in range(300):
        lstm_pred = model.predict(current_sequence)
        lstm_predicted_prices.append(lstm_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = lstm_pred
    
    lstm_predicted_prices = scaler.inverse_transform(np.array(lstm_predicted_prices).reshape(-1, 1))
    
    # Linear Regression
    X = np.arange(len(data_for_model)).reshape(-1, 1)
    lr_model = LinearRegression()
    lr_model.fit(X, data_for_model)
    
    X_predict = np.arange(len(data_for_model), len(data_for_model) + 300).reshape(-1, 1)
    lr_predicted_prices = lr_model.predict(X_predict)
    
    # Calculate MAE for LSTM and Linear Regression
    lstm_mae = mean_absolute_error(last_points[:300], lstm_predicted_prices)
    lr_mae = mean_absolute_error(last_points[:300], lr_predicted_prices)
    
    print(f"LSTM MAE: {lstm_mae:.2f}")
    print(f"Linear Regression MAE: {lr_mae:.2f}")
    
    plot_predictions(last_points, lstm_predicted_prices, lr_predicted_prices, lstm_mae, lr_mae)

def plot_predictions(actual, lstm_pred, lr_pred, lstm_mae, lr_mae):
    for widget in graph_frame.winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(df)-300, len(df)), actual, label='Actual Prices', color='green')
    ax.plot(range(len(df)-300, len(df)), lstm_pred, label=f'LSTM Predictions (MAE: {lstm_mae:.2f})', color='orange')
    ax.plot(range(len(df)-300, len(df)), lr_pred, label=f'Linear Regression Predictions (MAE: {lr_mae:.2f})', color='red')
    ax.set_title('BTC Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

root = tk.Tk()
root.title("Cryptocurrency Price Predictor")
root.configure(bg='#333')

style = ttk.Style(root)
style.theme_use('clam')
style.configure('TCombobox', fieldbackground='#333', background='#333', foreground='black')
style.configure('TButton', background='#555', foreground='white', borderwidth=1)
style.configure('TLabel', background='#333', foreground='white')
style.configure('TEntry', fieldbackground='#555', foreground='white')

ttk.Label(root, text="Select Coin:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
coin_combobox = ttk.Combobox(root, values=["BTC"], state="readonly", width=15)
coin_combobox.set("BTC")
coin_combobox.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(root, text="Select Currency:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
currency_combobox = ttk.Combobox(root, values=["USD"], state="readonly", width=15)
currency_combobox.set("USDT")
currency_combobox.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(root, text="Start Date:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
start_date_entry = ttk.Entry(root, width=15)
start_date_entry.grid(row=2, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(root, text="End Date:").grid(row=3, column=0, padx=10, pady=10, sticky='w')
end_date_entry = ttk.Entry(root, width=15)
end_date_entry.grid(row=3, column=1, padx=10, pady=10, sticky='ew')

load_data_button = ttk.Button(root, text="Load Data", command=load_data)
load_data_button.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky='w')

predict_data_button = ttk.Button(root, text="Predict Data", command=predict_data)
predict_data_button.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky='ew')

graph_frame = tk.Frame(root, bg='#555')
graph_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

root.grid_rowconfigure(6, weight=1)
root.grid_columnconfigure(1, weight=1)
start_date_entry.insert(0, '2024-01-24')
end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))

root.mainloop()
