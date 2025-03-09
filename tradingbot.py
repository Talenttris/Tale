from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import requests
import pandas as pd
import pandas_ta as ta
import os
import time
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ======== DERIV CONFIGURATION ========
DERIV_API_KEY = os.getenv('DERIV_API_KEY')
DERIV_APP_ID = os.getenv('DERIV_APP_ID')
DERIV_API_URL = "https://api.deriv.com"

SYMBOL_MAP = {
    # Volatility Indices
    "VOLATILITY100": "1HZ100V",
    "VOLATILITY75": "1HZ75V",
    "VOLATILITY50": "1HZ50V",
    "VOLATILITY25": "1HZ25V",
    
    # Boom & Crash Indices
    "BOOM500": "BOOM500",
    "BOOM1000": "BOOM1000",
    "CRASH500": "CRASH500", 
    "CRASH1000": "CRASH1000",
    
    # Step Indices
    "STEP": "STP",
    
    # Jump Indices
    "JUMP10": "JUMP10",
    "JUMP25": "JUMP25",
    "JUMP50": "JUMP50",
    "JUMP75": "JUMP75",
    "JUMP100": "JUMP100",
    
    # Forex Pairs
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "AUDUSD": "frxAUDUSD",
    "USDCAD": "frxUSDCAD",
    "USDCHF": "frxUSDCHF",
    "NZDUSD": "frxNZDUSD",
    
    # Commodities
    "XAUUSD": "frxXAUUSD",  # Gold
    "XAGUSD": "frxXAGUSD",  # Silver
    
    # Market Indices
    "WALLSTREET": "WLDAUD",
    "DOWJONES": "DJI",
    "NIKKEI225": "JP225",
    "FTSE100": "UK100"
}

TIMEFRAMES = {
    'analysis': '15m',
    'sl': '5m',
    'entry': '1m'
}
# =====================================

def convert_symbol(symbol):
    """Convert to Deriv's official symbol format"""
    return SYMBOL_MAP.get(symbol.upper(), symbol)

def get_deriv_data(symbol, timeframe):
    """Fetch historical data from Deriv API"""
    try:
        time.sleep(1)  # Rate limiting
        api_symbol = convert_symbol(symbol)
        url = f"{DERIV_API_URL}/market/candles"
        
        headers = {
            "Authorization": f"Bearer {DERIV_API_KEY}",
            "X-App-ID": DERIV_APP_ID,
            "Accept": "application/json"
        }

        params = {
            "symbol": api_symbol,
            "granularity": timeframe,
            "count": 200
        }

        response = requests.get(url, headers=headers, params=params)
        app.logger.info(f"API Request: {response.url}")
        
        if response.status_code != 200:
            app.logger.error(f"API Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        
        if 'error' in data:
            app.logger.error(f"Deriv API Error: {data['error']['message']}")
            return None

        # Create DataFrame
        df = pd.DataFrame(data['candles'])
        df = df.rename(columns={
            'epoch': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        })
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.set_index('time').astype(float)

    except Exception as e:
        app.logger.error(f"Data Error: {str(e)}")
        return None

def calculate_winrate(data):
    """Calculate historical winrate"""
    if len(data) < 100: return "N/A"
    changes = data['close'].pct_change().dropna()
    return f"{(len(changes[changes > 0])/len(changes))*100:.1f}%"

def determine_trend(data):
    """EMA Trend Detection"""
    if len(data) < 50: return "N/A"
    data['EMA20'] = ta.ema(data['close'], 20)
    data['EMA50'] = ta.ema(data['close'], 50)
    return "Up trend" if data['EMA20'].iloc[-1] > data['EMA50'].iloc[-1] else "Down trend"

def analyze_volatility(symbol):
    """Market analysis with Deriv-specific parameters"""
    try:
        # Get 15m data for trend
        df_15m = get_deriv_data(symbol, TIMEFRAMES['analysis'])
        if df_15m is None or len(df_15m) < 100:
            app.logger.error(f"Error: Insufficient 15m data for {symbol}")
            return None

        # Get 5m data for SL
        df_5m = get_deriv_data(symbol, TIMEFRAMES['sl'])
        if df_5m is None or len(df_5m) < 50:
            app.logger.error(f"Error: Insufficient 5m data for {symbol}")
            return None

        # Get 1m data for entry
        df_1m = get_deriv_data(symbol, TIMEFRAMES['entry'])
        if df_1m is None or len(df_1m) < 10:
            app.logger.error(f"Error: Insufficient 1m data for {symbol}")
            return None

        # Calculate ATRs
        df_5m['ATR'] = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], 14)
        df_15m['ATR'] = ta.atr(df_15m['high'], df_15m['low'], df_15m['close'], 14)
        atr_sl = df_5m['ATR'].iloc[-1]
        atr_tp = df_15m['ATR'].iloc[-1]

        # Support/Resistance
        support = df_15m['low'].rolling(50).min().iloc[-1]
        resistance = df_15m['high'].rolling(50).max().iloc[-1]
        last_close = df_1m['close'].iloc[-1]
        buffer = 0.005 * (resistance - support)

        # Signal detection
        trend = determine_trend(df_15m)
        signal = None
        
        if last_close > (resistance + buffer) and trend == "Up trend":
            signal = ('BUY', last_close, last_close - (atr_sl * 3))
        elif last_close < (support - buffer) and trend == "Down trend":
            signal = ('SELL', last_close, last_close + (atr_sl * 3))

        if not signal:
            app.logger.info(f"No signal: {symbol} at {last_close:.5f}")
            return None

        # Calculate targets
        direction, entry, sl = signal
        tp1 = entry + (atr_tp * 1) if direction == 'BUY' else entry - (atr_tp * 1)
        tp2 = entry + (atr_tp * 2) if direction == 'BUY' else entry - (atr_tp * 2)

        return {
            'symbol': symbol,
            'signal': direction,
            'winrate': calculate_winrate(df_15m),
            'trend': trend,
            'entry': round(entry, 5),
            'sl': round(sl, 5),
            'tp1': round(tp1, 5),
            'tp2': round(tp2, 5)
        }

    except Exception as e:
        app.logger.error(f"Analysis Error: {str(e)}")
        return None

@app.route("/")
def home():
    return "Deriv Multi-Asset Bot - Operational"

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.form.get("Body").strip().upper()
    response = MessagingResponse()

    if incoming_msg in ["HI", "HELLO", "START"]:
        response.message(
            "📈 Deriv Multi-Asset Bot 📈\n"
            "Supported Instruments:\n"
            "Volatility Indices:\n"
            "• VOLATILITY25/50/75/100\n"
            "Boom & Crash:\n"
            "• BOOM500/1000\n• CRASH500/1000\n"
            "Forex Pairs:\n"
            "• EURUSD • GBPUSD • USDJPY\n• AUDUSD • USDCAD • USDCHF • NZDUSD\n"
            "Commodities:\n• XAUUSD (Gold) • XAGUSD (Silver)\n"
            "Jump Indices:\n• JUMP10/25/50/75/100\n\n"
            "Commands:\n➤ Analysis: SYMBOL\n➤ Price: PRICE SYMBOL"
        )
        return str(response)

    if incoming_msg.startswith("PRICE "):
        symbol = incoming_msg.split(" ")[1]
        df = get_deriv_data(symbol, '1m')
        price = df['close'].iloc[-1] if df is not None else None
        response.message(f"Current {symbol}: {price:.5f}" if price else "❌ Price unavailable")
        return str(response)

    if incoming_msg in SYMBOL_MAP:
        symbol = incoming_msg
        analysis = analyze_volatility(symbol)

        if analysis:
            msg = (f"📊 {analysis['symbol']} Analysis\n"
                   f"Signal: {analysis['signal']}\n"
                   f"Winrate: {analysis['winrate']}\n"
                   f"15m Trend: {analysis['trend']}\n"
                   f"Entry: {analysis['entry']:.5f}\n"
                   f"SL: {analysis['sl']:.5f}\n"
                   f"TP1: {analysis['tp1']:.5f}\n"
                   f"TP2: {analysis['tp2']:.5f}")
        else:
            msg = f"No current opportunity in {symbol}"

        response.message(msg)
        return str(response)

    response.message("❌ Invalid command. Send 'HI' for help")
    return str(response)

@app.route("/debug/<symbol>")
def debug(symbol):
    try:
        api_symbol = convert_symbol(symbol)
        response = requests.get(
            f"{DERIV_API_URL}/market/candles",
            headers={
                "Authorization": f"Bearer {DERIV_API_KEY}",
                "X-App-ID": DERIV_APP_ID
            },
            params={"symbol": api_symbol, "granularity": "15m", "count": 1}
        )
        return jsonify({
            "symbol_mapping": f"{symbol} -> {api_symbol}",
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
