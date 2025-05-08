import pandas as pd
from src.envs.trading_env import TradingEnv
from src.strategy.strategy_rules import TrendFollowingStrategy
from src.utils.risk_manager import RiskManager

# Carregue o DataFrame de dados
csv_path = 'data/raw/ETH_USDT_1h.csv'  # Ajustado para novo diretório de dados
df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calc_rsi(df['close'])
df = df.dropna().reset_index(drop=True)

# Configuração da estratégia (ajuste conforme seu config)
from src.utils.config_loader import load_config

# Carrega configurações
trend_config = load_config('config/env_config.yaml')['trend_following']
strategy = TrendFollowingStrategy(
    bos_lookback=trend_config['entry_rules']['bos_lookback'],
    fvg_threshold=trend_config['entry_rules']['fvg_size']
)

risk_manager = RiskManager(
    risk_per_trade=0.01,
    rr_ratio=TREND_CONFIG['exit_rules']['risk_reward']
)

# Instancia o ambiente
env = TradingEnv(
    df=df,
    window_size=20,  # Valor padrão, ajuste conforme necessário
    initial_balance=10000,  # Saldo inicial para teste
    fee=0.001,  # Taxa de negociação
    symbols=['ETH'],  # Lista de símbolos
    strategy=strategy,
    risk_manager=risk_manager
)

obs = env.reset()
done = False
total_reward = 0
step = 0

print('Iniciando teste manual com agente aleatório...')
while not done:
    action = env.action_space.sample()  # ação aleatória
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1
    if step % 5000 == 0:
        print(f'Passo {step}: saldo={env.balance:.2f}, reward acumulado={total_reward:.2f}')

print('\n===== RESULTADO FINAL =====')
print(f'Saldo final: {env.balance:.2f}')
print(f'Lucro líquido: {env.balance - env.initial_balance:.2f}')
print(f'Reward total acumulado: {total_reward:.2f}')
print(f'Drawdown máximo: {env.max_drawdown:.2f}')
print(f'Número total de passos: {step}')
print('==========================')
