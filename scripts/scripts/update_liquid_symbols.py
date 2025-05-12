import ccxt
import yaml

def get_top_liquid_symbols(n=10, min_volume_usdt=10000000):
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    liquid = []
    for symbol, ticker in tickers.items():
        if '/USDT' in symbol and symbol in markets:
            vol = ticker.get('quoteVolume', 0)
            if vol and vol > min_volume_usdt:
                liquid.append((symbol.replace('/', ''), vol))
    # Ordena por volume decrescente
    liquid = sorted(liquid, key=lambda x: x[1], reverse=True)
    return [s for s, v in liquid[:n]]

def update_trade_config(symbols, config_path='/config/trade_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['trade']['symbols'] = symbols
    with open(config_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"Atualizado trade_config.yaml com os {len(symbols)} ativos mais líquidos.")

if __name__ == '__main__':
    top_symbols = get_top_liquid_symbols(n=10, min_volume_usdt=10000000)
    update_trade_config(top_symbols)