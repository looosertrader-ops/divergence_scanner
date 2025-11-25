"""
NSE Stock Divergence Scanner - Flask Web Application
Deploy this to PythonAnywhere, Heroku, or Render
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import io
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

# Global cache for stock lists and results
cache = {
    'fno_stocks': [],
    'last_fetch': None,
    'scan_results': None,
    'scanning': False,
    'progress': 0
}

def get_fno_stocks():
    """Fetches FnO stocks from NSE archives"""
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        print("Fetching official FnO list from NSE...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        df.columns = [c.strip().upper() for c in df.columns]
        
        if 'SYMBOL' not in df.columns:
            possible_cols = [c for c in df.columns if 'SYM' in c]
            if possible_cols:
                symbol_col = possible_cols[0]
            else:
                raise ValueError("Could not find SYMBOL column")
        else:
            symbol_col = 'SYMBOL'

        tickers = df[symbol_col].dropna().astype(str).str.strip().unique().tolist()
        
        indices_to_exclude = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SYMBOL']
        tickers = [t for t in tickers if t not in indices_to_exclude]
        
        formatted_tickers = [f"{t}.NS" for t in tickers]
        
        print(f"Successfully loaded {len(formatted_tickers)} FnO stocks.")
        return formatted_tickers, tickers

    except Exception as e:
        print(f"Error fetching FnO list: {e}")
        fallback = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 
                   'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 
                   'MARUTI', 'BAJFINANCE', 'HCLTECH', 'TITAN', 'SUNPHARMA']
        return [f"{t}.NS" for t in fallback], fallback

def calculate_rsi(series, period=14):
    """Calculates RSI using Wilder's Smoothing"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def scan_market(tickers):
    """Scans market for divergences"""
    print(f"Downloading data for {len(tickers)} stocks...")
    
    try:
        data = yf.download(tickers, period="3mo", group_by='ticker', progress=False, threads=True)
    except Exception as e:
        print(f"Download failed: {e}")
        return [], []

    bullish_stocks = []
    bearish_stocks = []
    total = len(tickers)
    
    print("\nAnalyzing divergences...")

    for idx, ticker in enumerate(tickers):
        try:
            cache['progress'] = int((idx / total) * 100)
            
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]:
                    continue
                df = data[ticker].copy()
            else:
                df = data.copy()

            df.dropna(subset=['Close', 'High', 'Low'], inplace=True)
            
            if len(df) < 30: 
                continue

            df['RSI'] = calculate_rsi(df['Close'], period=14)
            
            if len(df) < 15:
                continue

            current_close = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            past_high = df['High'].iloc[-15]
            past_low = df['Low'].iloc[-15]
            past_rsi = df['RSI'].iloc[-15]

            if np.isnan(current_rsi) or np.isnan(past_rsi):
                continue

            stock_name = ticker.replace('.NS', '')

            # Bearish Divergence
            if (current_close >= past_high) and (current_rsi < past_rsi):
                bearish_stocks.append({
                    'ticker': stock_name,
                    'segment': 'F&O',
                    'close': round(current_close, 2),
                    'high14': round(past_high, 2),
                    'rsi': round(current_rsi, 2),
                    'rsi14': round(past_rsi, 2),
                    'priceChange': round(((current_close - past_high) / past_high) * 100, 2),
                    'rsiChange': round(current_rsi - past_rsi, 2)
                })

            # Bullish Divergence
            elif (current_close < past_low) and (current_rsi > past_rsi):
                bullish_stocks.append({
                    'ticker': stock_name,
                    'segment': 'F&O',
                    'close': round(current_close, 2),
                    'low14': round(past_low, 2),
                    'rsi': round(current_rsi, 2),
                    'rsi14': round(past_rsi, 2),
                    'priceChange': round(((current_close - past_low) / past_low) * 100, 2),
                    'rsiChange': round(current_rsi - past_rsi, 2)
                })

        except Exception as e:
            continue

    cache['progress'] = 100
    return bullish_stocks, bearish_stocks

# API Endpoints
@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fetch-stocks', methods=['GET'])
def fetch_stocks():
    """Fetch FnO stock list from NSE"""
    try:
        formatted_tickers, raw_tickers = get_fno_stocks()
        cache['fno_stocks'] = formatted_tickers
        cache['last_fetch'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'count': len(formatted_tickers),
            'stocks': raw_tickers,
            'timestamp': cache['last_fetch']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scan', methods=['POST'])
def scan():
    """Start market scan"""
    if cache['scanning']:
        return jsonify({'error': 'Scan already in progress'}), 400
    
    def run_scan():
        cache['scanning'] = True
        cache['progress'] = 0
        
        if not cache['fno_stocks']:
            formatted_tickers, _ = get_fno_stocks()
            cache['fno_stocks'] = formatted_tickers
        
        bullish, bearish = scan_market(cache['fno_stocks'])
        
        cache['scan_results'] = {
            'bullishDivergence': bullish,
            'bearishDivergence': bearish,
            'totalScanned': len(cache['fno_stocks']),
            'timestamp': datetime.now().isoformat()
        }
        cache['scanning'] = False
        cache['progress'] = 100
    
    thread = threading.Thread(target=run_scan)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Scan started'})

@app.route('/api/scan-status', methods=['GET'])
def scan_status():
    """Get scan progress and results"""
    return jsonify({
        'scanning': cache['scanning'],
        'progress': cache['progress'],
        'results': cache['scan_results']
    })

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Stock Divergence Scanner</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .card { transition: all 0.3s; }
        .card:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0,0,0,0.3); }
        @keyframes spin { to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 1s linear infinite; }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-950 via-slate-900 to-slate-950 min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-2">NSE Stock Divergence Scanner</h1>
            <p class="text-slate-400">Automatic Detection of Bullish & Bearish Divergences</p>
        </div>

        <!-- Stock List Status -->
        <div class="bg-slate-800 rounded-lg shadow-xl p-4 mb-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div id="statusIndicator" class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span id="statusText" class="text-white">Loading...</span>
                </div>
                <button onclick="fetchStocks()" id="refreshBtn" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2">
                    <svg id="refreshIcon" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    Refresh List
                </button>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="bg-slate-800 rounded-lg shadow-xl p-6 mb-6">
            <button onclick="startScan()" id="scanBtn" class="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-slate-600 disabled:to-slate-600 text-white rounded-lg font-semibold flex items-center justify-center gap-2">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                </svg>
                Scan Market for Divergences
            </button>
            
            <!-- Progress Bar -->
            <div id="progressBar" class="hidden mt-4">
                <div class="flex justify-between text-sm text-slate-400 mb-2">
                    <span>Scanning...</span>
                    <span id="progressText">0%</span>
                </div>
                <div class="w-full bg-slate-700 rounded-full h-2">
                    <div id="progressFill" class="bg-blue-600 h-2 rounded-full transition-all" style="width: 0%"></div>
                </div>
            </div>

            <div class="mt-4 text-sm text-slate-400">
                <p><strong>Bearish:</strong> Price ≥ High (14d ago) but RSI &lt; RSI (14d ago)</p>
                <p><strong>Bullish:</strong> Price ≤ Low (14d ago) but RSI &gt; RSI (14d ago)</p>
            </div>
        </div>

        <!-- Results Summary -->
        <div id="summary" class="hidden bg-slate-800 rounded-lg p-4 mb-6 text-center">
            <p class="text-white text-lg">
                Scanned <strong id="totalScanned">0</strong> stocks | 
                Found <strong class="text-red-400" id="bearishCount">0</strong> Bearish & 
                <strong class="text-green-400" id="bullishCount">0</strong> Bullish Divergences
            </p>
        </div>

        <!-- Results Grid -->
        <div id="results" class="grid lg:grid-cols-2 gap-6 hidden">
            <!-- Bearish -->
            <div class="bg-red-900/20 border border-red-500/30 rounded-lg p-6">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <svg class="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path>
                        </svg>
                        <h2 class="text-2xl font-bold text-red-400">Bearish Divergence</h2>
                    </div>
                    <span class="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-semibold" id="bearishBadge">0</span>
                </div>
                <div id="bearishList" class="space-y-3 max-h-[600px] overflow-y-auto"></div>
            </div>

            <!-- Bullish -->
            <div class="bg-green-900/20 border border-green-500/30 rounded-lg p-6">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <svg class="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                        </svg>
                        <h2 class="text-2xl font-bold text-green-400">Bullish Divergence</h2>
                    </div>
                    <span class="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-semibold" id="bullishBadge">0</span>
                </div>
                <div id="bullishList" class="space-y-3 max-h-[600px] overflow-y-auto"></div>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-8 text-center text-slate-500 text-sm">
            <p><strong>Disclaimer:</strong> For educational purposes only. Not financial advice.</p>
            <p class="mt-1">Data from NSE & Yahoo Finance</p>
        </div>
    </div>

    <script>
        let stockCount = 0;

        async function fetchStocks() {
            const btn = document.getElementById('refreshBtn');
            const icon = document.getElementById('refreshIcon');
            const statusText = document.getElementById('statusText');
            const statusIndicator = document.getElementById('statusIndicator');
            
            btn.disabled = true;
            icon.classList.add('animate-spin');
            statusText.textContent = 'Fetching stock list from NSE...';
            statusIndicator.className = 'w-3 h-3 bg-yellow-500 rounded-full animate-pulse';
            
            try {
                const response = await fetch('/api/fetch-stocks');
                const data = await response.json();
                
                if (data.success) {
                    stockCount = data.count;
                    statusText.textContent = `Loaded: ${data.count} F&O stocks`;
                    statusIndicator.className = 'w-3 h-3 bg-green-500 rounded-full animate-pulse';
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                statusText.textContent = 'Error loading stocks: ' + error.message;
                statusIndicator.className = 'w-3 h-3 bg-red-500 rounded-full';
            } finally {
                btn.disabled = false;
                icon.classList.remove('animate-spin');
            }
        }

        async function startScan() {
            const scanBtn = document.getElementById('scanBtn');
            const progressBar = document.getElementById('progressBar');
            
            scanBtn.disabled = true;
            progressBar.classList.remove('hidden');
            
            try {
                const response = await fetch('/api/scan', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    pollScanStatus();
                }
            } catch (error) {
                alert('Error starting scan: ' + error.message);
                scanBtn.disabled = false;
                progressBar.classList.add('hidden');
            }
        }

        async function pollScanStatus() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch('/api/scan-status');
                    const data = await response.json();
                    
                    const progressFill = document.getElementById('progressFill');
                    const progressText = document.getElementById('progressText');
                    
                    progressFill.style.width = data.progress + '%';
                    progressText.textContent = data.progress + '%';
                    
                    if (!data.scanning && data.results) {
                        clearInterval(interval);
                        displayResults(data.results);
                        document.getElementById('scanBtn').disabled = false;
                        document.getElementById('progressBar').classList.add('hidden');
                    }
                } catch (error) {
                    clearInterval(interval);
                    console.error('Error polling status:', error);
                }
            }, 1000);
        }

        function displayResults(results) {
            document.getElementById('summary').classList.remove('hidden');
            document.getElementById('results').classList.remove('hidden');
            
            document.getElementById('totalScanned').textContent = results.totalScanned;
            document.getElementById('bearishCount').textContent = results.bearishDivergence.length;
            document.getElementById('bullishCount').textContent = results.bullishDivergence.length;
            document.getElementById('bearishBadge').textContent = results.bearishDivergence.length;
            document.getElementById('bullishBadge').textContent = results.bullishDivergence.length;
            
            const bearishList = document.getElementById('bearishList');
            const bullishList = document.getElementById('bullishList');
            
            bearishList.innerHTML = results.bearishDivergence.length === 0 
                ? '<p class="text-slate-400">No bearish divergence found</p>'
                : results.bearishDivergence.map(stock => `
                    <div class="card bg-slate-800 rounded-lg p-4 border border-slate-700 hover:border-red-500/50">
                        <div class="flex items-start justify-between mb-2">
                            <div class="font-bold text-xl text-white">${stock.ticker}</div>
                            <span class="px-2 py-1 rounded text-xs font-semibold bg-purple-600">${stock.segment}</span>
                        </div>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div class="text-slate-400">Close: <span class="text-white">₹${stock.close}</span></div>
                            <div class="text-slate-400">High (14d): <span class="text-white">₹${stock.high14}</span></div>
                            <div class="text-slate-400">RSI: <span class="text-white">${stock.rsi}</span></div>
                            <div class="text-slate-400">RSI (14d): <span class="text-white">${stock.rsi14}</span></div>
                            <div class="text-slate-400">Price: <span class="text-green-400">+${stock.priceChange}%</span></div>
                            <div class="text-slate-400">RSI: <span class="text-red-400">${stock.rsiChange}</span></div>
                        </div>
                    </div>
                `).join('');
            
            bullishList.innerHTML = results.bullishDivergence.length === 0
                ? '<p class="text-slate-400">No bullish divergence found</p>'
                : results.bullishDivergence.map(stock => `
                    <div class="card bg-slate-800 rounded-lg p-4 border border-slate-700 hover:border-green-500/50">
                        <div class="flex items-start justify-between mb-2">
                            <div class="font-bold text-xl text-white">${stock.ticker}</div>
                            <span class="px-2 py-1 rounded text-xs font-semibold bg-purple-600">${stock.segment}</span>
                        </div>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div class="text-slate-400">Close: <span class="text-white">₹${stock.close}</span></div>
                            <div class="text-slate-400">Low (14d): <span class="text-white">₹${stock.low14}</span></div>
                            <div class="text-slate-400">RSI: <span class="text-white">${stock.rsi}</span></div>
                            <div class="text-slate-400">RSI (14d): <span class="text-white">${stock.rsi14}</span></div>
                            <div class="text-slate-400">Price: <span class="text-red-400">${stock.priceChange}%</span></div>
                            <div class="text-slate-400">RSI: <span class="text-green-400">+${stock.rsiChange}</span></div>
                        </div>
                    </div>
                `).join('');
        }

        // Auto-fetch stocks on page load
        window.onload = fetchStocks;
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
