"""This is simplified version of the task1 without threading and queues"""
"""
f=open("op1.csv","a")
def on_message(ws, message):
    try:
        data = json.loads(message)
        ticker = data.get('data', [{}])[0].get('s')
        price = data.get('data', [{}])[0].get('p')
        quantity = data.get('data', [{}])[0].get('v')
        if ticker and price and quantity:
            f.write(f"{ticker},{price},{quantity}\n")
            f.flush()
    except Exception as e:
        pass
def on_error(ws, error):
    global f
    f.close()
    if isinstance(error, websocket._exceptions.WebSocketConnectionClosedException):
        time.sleep(5)
        ws.run_forever()
def on_open(ws):
    global tickers
    global f
    if not os.path.exists(OUTPUT_PATH):
        f = open(OUTPUT_PATH, 'a')
        f.write("ticker,price,volume\n")
    else:
        f = open(OUTPUT_PATH, 'a')
    f.flush()
    
    for ticker in tickers:
        ws.send(json.dumps({"type": "subscribe", "symbol": ticker}))
"""

"""
This module establishes a websocket connection to Finnhub.io to receive real-time stock data for specified tickers. 
It subscribes to the tickers, processes incoming messages, and saves the data (ticker, price, volume) to a CSV file.
The module utilizes the `websocket-client` library for websocket communication, `json` for parsing messages,
`logging` for recording events, `threading` for saving records in csv file in background, and `queue` for managing 
messages as buffer in case of data is arrived in high amount.
"""
import websocket
import json
import time
import os
import queue
import threading
import logging

# API_KEY = os.environ.get('API_KEY')
API_KEY = "csbo7m1r01qugk3m7co0csbo7m1r01qugk3m7cog"
# given here for testing this code

# Dictionary to store message queues for each ticker
ticker_queues = {}

# Tickers
tickers = ["MSFT", "AMZN"]

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='websocket_data.log',  # Log to this file
                    filemode='a')  # Append to the file
# CSV file handler
f=None

OUTPUT_PATH="op1.csv"


def on_message(ws, message):
    """
    Callback function when a message is received from the websocket.

    Args:
        ws: The websocket object.
        message: The message received.
    """
    # Parse the message to get the ticker symbol
    try:
        data = json.loads(message)
        ticker = data.get('data', [{}])[0].get('s')  # Assuming ticker is in 'data[0].s'
        if ticker:
            # Add message to the queue for the corresponding ticker
            if ticker in ticker_queues:
                ticker_queues[ticker].put(message)
            else:
                logging.warning(f"Received message for unknown ticker: {ticker}")
    except Exception as e:
          logging.error(f"Error parsing message: {e}\n")


def on_error(ws, error):
    """
    Callback function when an error occurs in the websocket.

    Args:
        ws: The websocket object.
        error: The error object.
    """
    global f
    # close the CSV file to prevent resource leakage
    f.close()
    # Reconnection logic : try to connect web socket after every 5 seconds
    if isinstance(error, websocket._exceptions.WebSocketConnectionClosedException):
        logging.error(f"WebSocket error: {error}\n")
        time.sleep(5)
        ws.run_forever()

def on_reconnect():
    """
    Callback function when the websocket reconnects.
    """
    global f
    logging.info("### reconnected ###\n")
    f=open(OUTPUT_PATH,'a')

def on_close(ws, close_status_code, close_msg):
    """
    Callback function when the websocket is closed.

    Args:
        ws: The websocket object.
        close_status_code: The close status code.
        close_msg: The close message.
    """
    global f
    logging.info("### closed ###\n")
    f.close()


def on_open(ws):
    """
    Callback function when the websocket is opened.

    Args:
        ws: The websocket object.
    """
    global tickers
    global ticker_queues
    global f
    if(not os.path.exists(OUTPUT_PATH)):
        f=open(OUTPUT_PATH,'a')
        f.write("ticker,price,volume\n")
    else:
        f=open(OUTPUT_PATH,'a')
    f.flush()
    logging.info("### opened ###")

    for ticker in tickers:
        ws.send(json.dumps({"type": "subscribe", "symbol": ticker}))
        logging.info(f"Subscribed to {ticker}")
        # Create a queue for the ticker if it doesn't exist
        if ticker not in ticker_queues:
            ticker_queues[ticker] = queue.Queue()
        # Start a thread for processing messages for the ticker in background
        threading.Thread(target=process_messages, args=(ticker,), daemon=True).start()

def process_messages(ticker:str):
    """
    Process messages for a specific ticker.

    Args:
        ticker: The ticker symbol.
    """
    global ticker_queues
    global f
    while True:
        try:
            message = ticker_queues[ticker].get(timeout=1)
            # Process the message for the specific ticker
            data = json.loads(message)
            ticker = data.get('data', [{}])[0].get('s')
            price = data.get('data', [{}])[0].get('p')
            quantity = data.get('data', [{}])[0].get('v')
            f.write(f"{ticker},{price},{quantity}\n")
            logging.info(f"Saving record for {ticker} into csv file")
            f.flush()
        except queue.Empty:
            # Handle empty queue (e.g., check connection, sleep)
            pass
        except Exception as e:
            logging.error(f"Error saving message: {e}")

if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={API_KEY}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close)
    ws.run_forever()
