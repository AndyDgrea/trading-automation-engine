# 🤖 Trading Automation Engine

An advanced Python-based trading automation system with real-time market analysis, signal generation, and automated execution via MetaTrader5 and Deriv API.

## ⚡ Features

- **Multi-Asset Support**: Trades across Volatility 10, 25, 50, 75, 100 indices (both regular and 1s variants)
- **Smart Pattern Detection**: Identifies bullish-to-bearish reversal patterns for precision entry
- **Real-time Analysis**: 
  - VWAP (Volume Weighted Average Price) calculation
  - RSI filtering for optimal entry points
  - Dynamic volatility and asset strength monitoring
- **Automated Execution**: Seamless integration with MT5 and Deriv API
- **Risk Management**: Built-in Martingale system with configurable stake levels
- **Instant Notifications**: 
  - Telegram alerts with trade signals and charts
  - Desktop notifications for critical events
  - Real-time trade status updates

## 🎯 Trading Strategy

The engine employs a sophisticated 3-candle pattern detection system:
1. **Pattern Recognition**: Identifies Bullish → Bearish → Bearish sequences
2. **Entry Timing**: Dynamic wait calculation for optimal entry
3. **Risk Filtering**: RSI thresholds prevent trades in oversold/overbought conditions
4. **Execution**: 57-second duration trades with immediate execution

## 📊 Technical Stack

- **Language**: Python 3.8+
- **Trading Platforms**: MetaTrader5, Deriv API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Communication**: Telegram Bot API, WebSockets
- **UI**: Tkinter (floating clock widget)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/AndyDgrea/trading-automation-engine.git
cd trading-automation-engine

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

1. **MetaTrader5 Credentials**:
   ```python
   account = YOUR_MT5_ACCOUNT
   password = "YOUR_PASSWORD"
   server = "YOUR_SERVER"
   ```

2. **Telegram Bot Setup**:
   ```python
   BOT_TOKEN = 'YOUR_BOT_TOKEN'
   CHAT_ID = 'YOUR_CHAT_ID'
   ```

3. **Deriv API Token**:
   ```python
   DERIV_TOKEN = "YOUR_DERIV_TOKEN"
   ```

## 🎮 Usage

```bash
# Run the trading bot
python trading_engine.py
```

The bot will:
- ✅ Initialize MT5 connection
- ✅ Subscribe to real-time market data
- ✅ Monitor all configured assets
- ✅ Generate signals based on pattern detection
- ✅ Execute trades automatically
- ✅ Send notifications for all events

## 🛡️ Risk Management

- **Martingale Levels**: Progressive stake system [500, 1200, 2800, 6200]
- **Stop Loss**: Automatic calculation based on pattern high/low
- **Signal Cooldown**: 2-minute cooldown between trades
- **RSI Filtering**: Prevents entries in extreme market conditions

## 📈 Performance Features

- **Time Synchronization**: MT5-Deriv offset calculation for precise timing
- **WebSocket Integration**: Low-latency trade execution
- **Multi-threaded Design**: Concurrent monitoring and execution
- **Error Handling**: Robust retry mechanisms and fallback systems

## 🔔 Notification System

- **Trade Signals**: Detailed entry, stop loss, and RSI levels
- **Charts**: Auto-generated candlestick charts with pattern highlights
- **Trade Results**: Win/loss tracking with profit calculations
- **System Status**: Connection status and error alerts

## 📱 Features in Detail

### Pattern Detection
- Analyzes 3-candle sequences
- Validates with VWAP trend
- Confirms with RSI levels
- Highlights patterns on charts

### Execution System
- Dynamic entry timing (N-1 second wait)
- Immediate Martingale recovery
- Contract status monitoring
- Profit/loss tracking

### Monitoring Tools
- Floating clock widget
- Real-time offset tracking
- Market volatility analysis
- Asset strength indicators

## 🔧 Dependencies

See [requirements.txt](requirements.txt) for full list.

Key libraries:
- `MetaTrader5` - MT5 API integration
- `websocket-client` - Deriv WebSocket connection
- `pandas/numpy` - Data processing
- `matplotlib` - Chart generation
- `plyer` - Desktop notifications

## 📝 Notes

- Ensure MT5 is running and logged in before starting
- Test with demo account first
- Monitor initial trades closely
- Adjust RSI thresholds per asset volatility
- Review Telegram notifications for all signals

## ⚠️ Disclaimer

**This software is for educational purposes only.** Trading involves substantial risk of loss. Past performance does not guarantee future results. Always:
- Test thoroughly with demo accounts
- Start with minimal stake amounts
- Monitor trades actively
- Understand the risks involved
- Never trade with money you cannot afford to lose

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

**Andrew Ewuola**
- Email: andrewewuola@gmail.com
- Portfolio: [andrew-ewuola.netlify.app](https://andrew-ewuola.netlify.app)
- GitHub: [@AndyDgrea](https://github.com/AndyDgrea)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⭐ **Star this repo if you find it useful!**

Built with 🔥 by Andrew Ewuola | Trading Systems Engineer
