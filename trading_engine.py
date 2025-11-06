import MetaTrader5 as mt5
import time
import requests
from plyer import notification
import pandas as pd
import signal
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import threading
import json
import websocket
import queue
import logging
import tkinter as tk
import numpy as np
import random
from collections import deque

# === Setup logging ===
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Telegram Setup ===
BOT_TOKEN = 'add token'
CHAT_ID = 'add chat id'

telegram_session = requests.Session()

# === Trading Assets and Config ===
# === Trading Assets ===
ASSETS = [
    "Volatility 10 Index",
    "Volatility 25 Index",
    "Volatility 50 Index",
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Volatility 10 (1s)",  # Added Volatility 10 (1s)
    "Volatility 25 (1s)",  # Added Volatility 25 (1s)
    "Volatility 50 (1s)",  # Added Volatility 50 (1s)
    "Volatility 75 (1s)",  # Added Volatility 75 (1s)
    "Volatility 100 (1s)"  # Added Volatility 100 (1s)
]

# Deriv symbols for the trading assets
DERIV_SYMBOLS = {
    "Volatility 10 Index": "R_10",
    "Volatility 25 Index": "R_25",
    "Volatility 50 Index": "R_50",
    "Volatility 75 Index": "R_75",
    "Volatility 100 Index": "R_100",
    "Volatility 10 (1s)": "R_10_1s",  # Added corresponding symbol for Volatility 10 (1s)
    "Volatility 25 (1s)": "R_25_1s",  # Added corresponding symbol for Volatility 25 (1s)
    "Volatility 50 (1s)": "R_50_1s",  # Added corresponding symbol for Volatility 50 (1s)
    "Volatility 75 (1s)": "R_75_1s",  # Added corresponding symbol for Volatility 75 (1s)
    "Volatility 100 (1s)": "R_100_1s"  # Added corresponding symbol for Volatility 100 (1s)
}

ASSET_CONFIG = {
    "Volatility 10 Index": {
        "rsi_low": 30,
        "rsi_high": 70,
    },
    "Volatility 25 Index": {
        "rsi_low": 25,
        "rsi_high": 75,
    },
    "Volatility 50 Index": {
        "rsi_low": 30,
        "rsi_high": 70,
    },
    "Volatility 75 Index": {
        "rsi_low": 35,
        "rsi_high": 65,
    },
    "Volatility 100 Index": {
        "rsi_low": 40,
        "rsi_high": 60,
    }
}

# Global dict to track last sent signal per symbol
last_signals = {}

offset_lock = threading.Lock()
offset_history = []
MAX_HISTORY = 10
mt5_deriv_offset = 0.0
latest_deriv_candle_close = None
candle_duration = 60

signal_queue = queue.Queue()

last_signal_time = None
last_message_time = None
signal_cooldown = 2 * 60
patience_message_interval = 10 * 60

# Martingale stack
TRADE_STACK = [500, 1200, 2800, 6200]
def get_market_data(symbol):
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        return None
    return tick_info

def calculate_volatility(symbol, timeframe=mt5.TIMEFRAME_M1, bars=100):
    try:
        # Get the last `bars` number of rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return 0.0001  # Default value if no data
        
        df = pd.DataFrame(rates)
        # Calculate the price range (high - low) for each bar and the average range
        df['price_range'] = df['high'] - df['low']
        volatility = df['price_range'].mean()  # average price range as a measure of volatility
        return volatility
    except Exception as e:
        print(f"Error calculating volatility for {symbol}: {e}")
        return 0.0001  # Default value on error

def calculate_asset_strength(symbol, timeframe=mt5.TIMEFRAME_M1, bars=100):
    try:
        # Calculate the average volume for the past `bars` number of rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return 50.0  # Default value if no data
        
        df = pd.DataFrame(rates)
        average_volume = df['real_volume'].mean()
        latest_volume = df['real_volume'].iloc[-1]
        
        # Avoid division by zero
        if average_volume <= 0:
            return 50.0  # Default value
            
        asset_strength = (latest_volume / average_volume) * 100  # Percent ratio of current volume to average
        return asset_strength
    except Exception as e:
        print(f"Error calculating asset strength for {symbol}: {e}")
        return 50.0  # Default value on error

def generate_market_info(symbol):
    try:
        tick_info = get_market_data(symbol)
        if not tick_info:
            return "Error: Could not fetch market data"

        # Get volatility and asset strength by volume
        volatility = calculate_volatility(symbol)
        asset_strength = calculate_asset_strength(symbol)
        
        # Simulate volume result and sentiment
        volume_result = random.randint(20, 60)  # Placeholder for real calculation
        sentiment = "Upward pressure" if random.random() > 0.5 else "Downward pressure"

        # Convert volatility to a descriptive term
        if volatility > 0.0002:
            volatility_desc = "Increased"
        else:
            volatility_desc = "Stable"

        market_info = f"""📡 *Market info:*
*• Volatility:* {volatility_desc}
*• Asset strength by volume:* {round(asset_strength, 2)}%
*• Volume result:* {volume_result}%
*• Sentiment:* {sentiment}
"""
        return market_info
    except Exception as e:
        print(f"Error generating market info for {symbol}: {e}")
        return "Market info unavailable"

def is_doji(candle):
    return abs(candle['close'] - candle['open']) <= 0.1 * (candle['high'] - candle['low'])

def calculate_vwap(df):
    try:
        q = df['tick_volume']
        p = (df['high'] + df['low'] + df['close']) / 3
        vwap = (p * q).cumsum() / q.cumsum()
        return vwap
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        # Return a default series with the same index as df
        return pd.Series(df['close'].values, index=df.index)

# New function to calculate VWAP slope
def calculate_vwap_slope(df):
    try:
        # Calculate the slope as the difference between the last two VWAP values
        return df['VWAP'].iloc[-1] - df['VWAP'].iloc[-2]
    except Exception as e:
        print(f"Error calculating VWAP slope: {e}")
        return 0

def detect_bullish_to_bearish_pattern(df):
    """
    Detects a 3-candle pattern where:
    - Candle n: Bullish (close > open)
    - Candle n+1: Bearish (close < open)  
    - Candle n+2: Bearish (close < open)
    
    Returns:
        tuple: (signal_found, signal_type, entry_price, stop_loss) or (False, None, None, None)
    """
    try:
        # Ensure we have at least 3 candles
        if len(df) < 3:
            return False, None, None, None
        
        # Get the last 3 candles
        candle_n = df.iloc[-3]    # Third last candle
        candle_n1 = df.iloc[-2]   # Second last candle  
        candle_n2 = df.iloc[-1]   # Last candle
        
        # Check if pattern matches:
        # n: bullish (close > open)
        # n+1: bearish (close < open)
        # n+2: bearish (close < open)
        
        is_bullish_n = candle_n['close'] > candle_n['open']
        is_bearish_n1 = candle_n1['close'] < candle_n1['open']
        is_bearish_n2 = candle_n2['close'] < candle_n2['open']
        
        if is_bullish_n and is_bearish_n1 and is_bearish_n2:
            # Pattern found - generate sell signal
            entry_price = candle_n2['close']  # Enter at close of 3rd candle
            stop_loss = candle_n['high']      # Stop above highest point of bullish candle
            
            return True, "Bullish-to-Bearish Reversal", entry_price, stop_loss
        
        return False, None, None, None
        
    except Exception as e:
        print(f"Error detecting pattern: {e}")
        return False, None, None, None

def check_strategy_conditions(symbol):
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) < 3:
            return False, pd.DataFrame(), "", None, None
            
        df = pd.DataFrame(rates)
        df['VWAP'] = calculate_vwap(df)

        if len(df) < 3:
            return False, df, "", None, None

        # Check for the bullish-to-bearish pattern
        pattern_found, signal_type, entry_price, stop_loss = detect_bullish_to_bearish_pattern(df)
        
        if pattern_found:
            return "sell", df, signal_type, entry_price, stop_loss
        
        return False, df, "", None, None
        
    except Exception as e:
        print(f"Error checking strategy for {symbol}: {e}")
        return False, pd.DataFrame(), "", None, None

# === Main Trading Logic ===
def check_and_trade_asset(symbol):
    global last_signal_time, last_signals

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
    if rates is None or len(rates) < 10:
        logger.warning(f"Insufficient data for {symbol}")
        return False

    df = pd.DataFrame(rates)
    df['VWAP'] = calculate_vwap(df)

    # Use new candlestick pattern detection for all symbols
    pattern_found, signal_type, entry_price, stop_loss = detect_bullish_to_bearish_pattern(df)
    
    if not pattern_found:
        logger.info(f"{symbol}: No bullish-to-bearish pattern detected.")
        return False

    # Only sell signals are generated by this pattern
    chosen_direction = "sell"
    chosen_signal_type = signal_type
    chosen_df = df

    # Check for duplicate signals
    last_signal = last_signals.get(symbol, None)
    if last_signal:
        if last_signal["type"] == chosen_signal_type and last_signal["direction"] == chosen_direction:
            logger.info(f"Skipped duplicate signal on {symbol} - Type: {chosen_signal_type}, Direction: {chosen_direction}")
            return False

    # Apply RSI filter - only trade if RSI is not oversold
    config = ASSET_CONFIG.get(symbol, {})
    rsi_low = config.get("rsi_low", 30)
    current_rsi = df['RSI'].iloc[-1]
    
    if current_rsi <= rsi_low:
        logger.info(f"{symbol}: RSI filter blocked signal - RSI={current_rsi:.2f} <= {rsi_low}")
        return False

    # Compose Telegram message
    msg = f"✅ Signal Detected: *{chosen_direction.upper()}* for *{symbol}* - Trading Immediately!\n"
    msg += f"📡 Signal Type: {chosen_signal_type}\n"
    msg += f"💰 Entry Price: {entry_price:.5f}\n"
    msg += f"🛑 Stop Loss: {stop_loss:.5f}\n"
    msg += f"📊 RSI: {current_rsi:.2f}"

    send_telegram_message(msg)
    send_telegram_chart(symbol, chosen_df, chosen_direction)
    send_desktop_notification(f"Trade Signal", f"{chosen_direction.upper()} for {symbol}")

    last_signals[symbol] = {"type": chosen_signal_type, "direction": chosen_direction}
    last_signal_time = time.time()

    send_trade_to_deriv(chosen_direction, symbol)
    return True

# === Telegram messaging and chart sending ===
def send_telegram_message(message, max_retries=3):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    for attempt in range(max_retries):
        try:
            response = telegram_session.post(url, data=payload)
            response.raise_for_status()
            logger.info(f"Telegram message sent: {message[:50]}...")
            break
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed to send Telegram message: {e}")
            time.sleep(2)
    else:
        logger.error("Failed to send Telegram message after retries")

def send_telegram_chart(symbol, df, signal_type, max_retries=3):
    recent = df.tail(30).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        # Plot candlesticks
        for i in range(len(recent)):
            candle = recent.iloc[i]
            color = 'green' if candle['close'] > candle['open'] else 'red'
            ax.plot([i, i], [candle['low'], candle['high']], color='black', linewidth=1)
            ax.add_patch(plt.Rectangle((i - 0.3, min(candle['open'], candle['close'])),
                                       0.6, abs(candle['open'] - candle['close']),
                                       color=color, alpha=0.8))

        # Highlight the pattern candles (last 3)
        if len(recent) >= 3:
            for i in range(len(recent)-3, len(recent)):
                candle = recent.iloc[i]
                ax.add_patch(plt.Rectangle((i - 0.4, candle['low']),
                                           0.8, candle['high'] - candle['low'],
                                           fill=False, edgecolor='yellow', linewidth=2))

        # Plot VWAP
        if 'VWAP' in df.columns:
            ax.plot(range(len(recent)), recent['VWAP'].values, color='blue', label='VWAP', linewidth=2)
        
        # Add RSI info to title
        current_rsi = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 0
        ax.set_title(f"{symbol} - {signal_type.upper()} Signal (RSI: {current_rsi:.1f})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        files = {'photo': buf}
        payload = {
            'chat_id': CHAT_ID,
            'caption': f"{symbol.upper()} {signal_type.upper()} signal chart\n⏲ {datetime.datetime.now().strftime('%H:%M:%S')}\n📊 Pattern: Bullish → Bearish → Bearish"
        }

        for attempt in range(max_retries):
            response = telegram_session.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", files=files, data=payload)
            if response.status_code == 200:
                logger.info(f"Chart sent successfully for {symbol} {signal_type}")
                break
            else:
                logger.warning(f"Attempt {attempt + 1} failed to send chart: {response.status_code} {response.text}")
                time.sleep(2)
        else:
            logger.error("Failed to send chart after retries")

    except Exception as e:
        logger.error(f"Exception in send_telegram_chart: {e}")
    finally:
        buf.close()
        plt.close(fig)

def send_desktop_notification(title, message):
    try:
        notification.notify(title=title, message=message, timeout=10)
        logger.info(f"Desktop notification sent: {title}")
    except Exception as e:
        logger.error(f"Failed to send desktop notification: {e}")

# === MT5 Initialization ===
account = mt5 account id
password = "mt5 password"
server = "mt5 server"

def initialize_mt5_with_retry(max_attempts=3, delay=5):
    for attempt in range(max_attempts):
        if mt5.initialize(login=account, password=password, server=server):
            logger.info("MT5 initialized successfully")
            send_telegram_message("✅ MT5 Login Successful!")
            return True
        else:
            err = mt5.last_error()
            logger.error(f"MT5 init attempt {attempt + 1} failed: {err}")
            time.sleep(delay)
    send_telegram_message("⚠️ MT5 Login failed after multiple attempts!")
    return False

# === WebSocket and Trade Handling ===
DERIV_TOKEN = "deriv api token"

trade_completed_event = threading.Event()

ws_app = None
ws_connected = False
buy_sent = False
current_level = 0
trade_result = None
current_direction = None
current_symbol = None
trade_results_history = []

def get_contract_type(direction):
    return "CALL" if direction.lower() == "buy" else "PUT"

def on_open(ws):
    global ws_connected
    logger.info("[WS] Connection opened. Authorizing...")
    ws_connected = True
    auth_payload = {
        "authorize": DERIV_TOKEN
    }
    ws.send(json.dumps(auth_payload))
    logger.info("[WS] Authorization request sent.")

def on_message(ws, message):
    global current_level, trade_result, ws_connected, buy_sent, trade_results_history

    try:
        data = json.loads(message)
        msg_type = data.get("msg_type")
        logger.info(f"[WS] Received message type: {msg_type}")

        if msg_type == "authorize":
            if "error" in data:
                logger.error(f"[WS] Authorization failed: {data['error']['message']}")
                send_telegram_message(f"❌ Deriv authorization failed: {data['error']['message']}")
                trade_completed_event.set()
                return

            logger.info("[WS] Authorized successfully.")

        elif msg_type == "proposal":
            if "error" in data:
                logger.error(f"[WS] Proposal error: {data['error']['message']}")
                send_telegram_message(f"❌ Proposal error: {data['error']['message']}")
                trade_completed_event.set()
                return

            proposal_id = data["proposal"]["id"]
            logger.info(f"[WS] Proposal received with ID: {proposal_id}. Sending buy request immediately...")
            buy_request = {
                "buy": proposal_id,
                "price": TRADE_STACK[current_level]
            }
            ws.send(json.dumps(buy_request))
            buy_sent = True

        elif msg_type == "buy":
            if "error" in data:
                logger.error(f"[WS] Buy error: {data['error']['message']}")
                send_telegram_message(f"❌ Buy error: {data['error']['message']}")
                trade_completed_event.set()
                return

            contract_id = data["buy"]["contract_id"]
            level_text = "Entry" if current_level == 0 else f"Martingale Level {current_level}"
            logger.info(f"[WS] Trade placed! {level_text} | Contract ID: {contract_id}")

            if current_level == 0:
                time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                send_telegram_message(
                    f"✅ Trade placed for *{current_direction.upper()}* on *{current_symbol}*\n"
                    f"💰 Amount: ${TRADE_STACK[current_level]}\n"
                    f"📊 Level: *{level_text}*\n"
                    f"⏱️ Execution Time: *{time_str}*\n"
                    f"🆔 Contract ID: `{contract_id}`"
                )

            ws.send(json.dumps({"forget_all": "proposal"}))
            ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }))

        elif msg_type == "proposal_open_contract":
            if "error" in data:
                logger.error(f"[WS] Contract monitoring error: {data['error']['message']}")
                trade_completed_event.set()
                return

            contract_info = data["proposal_open_contract"]
            if contract_info.get("is_expired") or contract_info.get("is_settleable") or contract_info.get("status") == "sold":
                profit = float(contract_info.get("profit", 0))
                subscription_id = data.get("subscription", {}).get("id")
                if subscription_id:
                    ws.send(json.dumps({"forget": subscription_id}))

                trade_result = profit > 0
                level_text = "Entry" if current_level == 0 else f"Martingale Level {current_level}"

                trade_results_history.append({
                    "level": current_level,
                    "level_text": level_text,
                    "profit": profit,
                    "win": trade_result
                })

                if trade_result:
                    logger.info(f"[WS] ✅ Win on {level_text}! Profit: ${profit:.2f}")

                    results_message = f"🎯 *Trade Result ({current_symbol}):* ✅ Win on {level_text}!\n💰 Profit: ${profit:.2f}"

                    if len(trade_results_history) > 1:
                        results_message += "\n\n📊 *Trade Sequence:*"
                        total_profit = 0
                        for i, result in enumerate(trade_results_history):
                            total_profit += result["profit"]
                            result_icon = "✅" if result["win"] else "❌"
                            results_message += f"\n{result_icon} {result['level_text']}: ${result['profit']:.2f}"
                        results_message += f"\n\n💵 *Net Result:* ${total_profit:.2f}"

                    send_telegram_message(results_message)

                    current_level = 0
                    trade_results_history.clear()
                    global last_signal_time
                    last_signal_time = time.time()
                    trade_completed_event.set()
                else:
                    logger.info(f"[WS] ❌ Loss on {level_text}. Loss: ${abs(profit):.2f}")
                    next_level = current_level + 1

                    if next_level < len(TRADE_STACK):
                        logger.info(f"[WS] Proceeding to Martingale Level {next_level} immediately...")
                        current_level = next_level
                        trade_completed_event.set()
                    else:
                        logger.info("[WS] Max Martingale level reached. Reporting all results.")

                        results_message = f"🎯 *Trade Sequence Results ({current_symbol}):*\n"
                        total_profit = 0
                        for i, result in enumerate(trade_results_history):
                            total_profit += result["profit"]
                            result_icon = "✅" if result["win"] else "❌"
                            results_message += f"\n{result_icon} {result['level_text']}: ${result['profit']:.2f}"

                        results_message += f"\n\n💵 *Net Result:* ${total_profit:.2f}"
                        results_message += "\n⚠️ Max Martingale level reached."

                        send_telegram_message(results_message)

                        current_level = 0
                        trade_results_history.clear()
                        last_signal_time = time.time()
                        trade_completed_event.set()

    except Exception as e:
        logger.error(f"[WS] Error processing message: {e}")
        send_telegram_message(f"⚠️ Error processing trade message: {str(e)[:100]}")
        trade_completed_event.set()

def on_error(ws, error):
    logger.error(f"[WS][Error] {error}")
    send_telegram_message(f"⚠️ WebSocket error: {error}")
    trade_completed_event.set()

def on_close(ws, close_status_code, close_msg):
    global ws_connected
    ws_connected = False
    logger.info(f"[WS] Connection closed: {close_status_code} - {close_msg}")
    trade_completed_event.set()

def send_trade_to_deriv(direction, symbol, is_martingale=False):
    global current_direction, current_symbol, trade_completed_event, ws_app, ws_connected, current_level, trade_result, buy_sent, trade_results_history

    if not is_martingale:
        trade_results_history.clear()

    current_direction = direction
    current_symbol = symbol
    trade_completed_event.clear()
    buy_sent = False

    if ws_app and ws_connected:
        logger.info("[WS] Closing existing connection before starting new trade")
        ws_app.close()
        time.sleep(1)

    logger.info("[WS] Creating new connection for trade")
    websocket.enableTrace(True)
    ws_app = websocket.WebSocketApp(
        "wss://ws.binaryws.com/websockets/v3?app_id=75952",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    connection_timeout = 10
    connection_start = time.time()
    while not ws_connected and time.time() - connection_start < connection_timeout:
        time.sleep(0.5)

    if not ws_connected:
        logger.error("[WS] Failed to establish connection within timeout")
        send_telegram_message("⚠️ Failed to establish connection to Deriv API")
        trade_completed_event.set()
        return

    if not is_martingale:
        now = datetime.datetime.now()
        seconds = now.second
        Y = 60 - seconds
        N = Y - 1 # dynamic wait before entry trade
        if N < 0:
            N += 60
        logger.info(f"Dynamic wait before sending entry trade: {N} seconds (signal second: {seconds})")
        time.sleep(N)
        logger.info(f"Waited {N} seconds, sending entry trade request now!")
    else:
        logger.info("Martingale trade - executing immediately without waiting")

    duration = 57
    duration_unit = "s"

    proposal_request = {
        "proposal": 1,
        "amount": TRADE_STACK[current_level],
        "basis": "stake",
        "contract_type": get_contract_type(current_direction),
        "currency": "USD",
        "duration": duration,
        "duration_unit": duration_unit,
        "symbol": DERIV_SYMBOLS[current_symbol]
    }
    ws_app.send(json.dumps(proposal_request))
    logger.info(f"Proposal request sent at {datetime.datetime.utcnow()}")

    if not trade_completed_event.wait(timeout=120):
        logger.warning("[WS] Trade timeout: no confirmation received within 120 seconds.")
        send_telegram_message("⚠️ Trade timeout: no confirmation received.")

    if ws_app and ws_connected:
        logger.info("[WS] Closing websocket after trade completion or timeout")
        ws_app.close()
        time.sleep(1)

    if not trade_result and current_level > 0 and current_level < len(TRADE_STACK):
        logger.info(f"Martingale level {current_level} immediate retry")
        send_trade_to_deriv(direction, symbol, is_martingale=True)

# === MT5 time sync and Deriv candle websocket ===

def fetch_mt5_server_time():
    try:
        symbol = ASSETS[0]
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is not None and len(rates) > 0:
            last_bar = rates[0]
            timestamp = last_bar['time']
            return datetime.datetime.utcfromtimestamp(timestamp)
        else:
            logger.warning("No MT5 rates for time fetch")
            return None
    except Exception as e:
        logger.error(f"Failed to get MT5 server time: {e}")
        return None

latest_deriv_candle_close = None

def deriv_candle_ws_loop():
    ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=75952"

    def on_open(ws):
        logger.info("Deriv candle WS open, subscribing")
        subscribe_to_deriv_candles(ws, DERIV_SYMBOLS[ASSETS[0]])

    def on_message(ws, message):
        global latest_deriv_candle_close
        data = json.loads(message)
        if 'candles' in data:
            candles = data['candles']
            if candles:
                last_candle = candles[-1]
                latest_deriv_candle_close = last_candle['epoch'] + 60
                logger.info(f"Latest Deriv candle close at {datetime.datetime.utcfromtimestamp(latest_deriv_candle_close)}")

    def on_error(ws, error):
        logger.error(f"Deriv candle WS error: {error}")

    def on_close(ws, code, reason):
        logger.info(f"Deriv candle WS closed: {code} {reason}")

    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()

def subscribe_to_deriv_candles(ws, symbol):
    req = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 50,
        "end": "latest",
        "style": "candles",
        "subscribe": 1,
        "granularity": 60
    }
    ws.send(json.dumps(req))

mt5_deriv_offset = 0.0
offset_history = []
MAX_HISTORY = 10
offset_lock = threading.Lock()

def update_offset_loop():
    global mt5_deriv_offset
    while True:
        mt5_time = fetch_mt5_server_time()
        if mt5_time and latest_deriv_candle_close:
            deriv_time_dt = datetime.datetime.utcfromtimestamp(latest_deriv_candle_close)
            offset = (mt5_time - deriv_time_dt).total_seconds()
            with offset_lock:
                offset_history.append(offset)
                if len(offset_history) > MAX_HISTORY:
                    offset_history.pop(0)
                mt5_deriv_offset = sum(offset_history) / len(offset_history)
                logger.info(f"Updated MT5-Deriv offset: {mt5_deriv_offset:.2f} seconds")
        else:
            logger.warning("Waiting for both MT5 time and Deriv candle time to calculate offset")
        time.sleep(30)

# === Floating Clock widget ===
def start_clock():
    root = tk.Tk()
    root.title("Clock Widget")
    root.geometry("150x50")
    root.attributes("-topmost", True)
    root.overrideredirect(True)

    label = tk.Label(root, font=("Helvetica", 30), fg="white", bg="black")
    label.pack(fill="both", expand=True)

    x_start = y_start = 0

    def update_clock():
        now = time.strftime("%H:%M:%S")
        label.config(text=now)
        label.after(1000, update_clock)

    def start_move(event):
        nonlocal x_start, y_start
        x_start = event.x
        y_start = event.y

    def do_move(event):
        x = event.x_root - x_start
        y = event.y_root - y_start
        root.geometry(f"+{x}+{y}")

    root.bind("<ButtonPress-1>", start_move)
    root.bind("<B1-Motion>", do_move)

    update_clock()
    root.mainloop()

# === Bot control and main loop ===
def run_bot():
    global last_signal_time, last_message_time

    if last_signal_time and time.time() - last_signal_time < signal_cooldown:
        remaining = int(signal_cooldown - (time.time() - last_signal_time))
        if not last_message_time or time.time() - last_message_time > patience_message_interval:
            mins = remaining // 60
            secs = remaining % 60
            logger.info(f"Waiting for cooldown... {mins}m {secs}s remaining")
            last_message_time = time.time()
        return

    if not signal_queue.empty():
        return

    for symbol in ASSETS:
        logger.info(f"Checking {symbol}...")
        trade_placed = check_and_trade_asset(symbol)
        if trade_placed:
            return

def stop_bot(signum, frame):
    logger.info("Bot stopping...")
    send_telegram_message("🛑 Bot stopped.")
    if ws_app:
        logger.info("Closing websocket connection...")
        ws_app.close()
    logger.info("Shutting down MT5...")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, stop_bot)
signal.signal(signal.SIGTERM, stop_bot)

def main():
    if not initialize_mt5_with_retry():
        logger.error("Failed to initialize MT5 after retries. Exiting.")
        sys.exit(1)

    logger.info(f"[✅] Bot is running on multiple assets: {', '.join(ASSETS)}...")
    send_telegram_message(f"🤖 Bot active on assets: {', '.join(ASSETS)}\n⚡ IMMEDIATE TRADING ENABLED - 57s duration trades!")

    global current_direction, trade_result, last_signal_time, last_message_time
    current_direction = None
    trade_result = None
    last_signal_time = None
    last_message_time = None

    deriv_candle_thread = threading.Thread(target=deriv_candle_ws_loop, daemon=True)
    deriv_candle_thread.start()

    offset_thread = threading.Thread(target=update_offset_loop, daemon=True)
    offset_thread.start()

    clock_thread = threading.Thread(target=start_clock, daemon=True)
    clock_thread.start()

    try:
        while True:
            now = datetime.datetime.now()
            if now.second == 0:
                run_bot()
            time.sleep(1)
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        send_telegram_message(f"⚠️ Bot encountered an error: {str(e)[:100]}")
        stop_bot(None, None)

if __name__ == "__main__":
    main()
