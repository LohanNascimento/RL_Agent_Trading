import pandas as pd
import numpy as np
import logging
import os
import sys
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler

from src.envs.trading_env import TradingEnv
from src.utils.technical_indicators import add_technical_indicators
from src.strategy.strategy_rules import TrendFollowingStrategy, ReversalStrategy, ScalpMomentumStrategy
from src.utils.risk_manager import RiskManager, RiskParameters
from src.utils.config_loader import load_config
from src.utils.train_pipeline import prepare_data, normalize_features, train_model

# Configuração de logging padronizada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%H:%M:%S'
)

# Função para limpar valores negativos em um DataFrame
def fix_negative_values(df, stage_name):
    negative_cols = []
    for col in df.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_cols.append((col, neg_count))
            # Tratar valores negativos para evitar problemas
            if col in ['close', 'open', 'high', 'low', 'volume']:
                # Para preços e volume, substituir negativos por um valor pequeno positivo
                df.loc[df[col] < 0, col] = 0.0001
            elif col in ['macd', 'macd_signal']:
                # MACD pode ser negativo naturalmente, não precisa tratar
                pass
            else:
                # Para outros indicadores técnicos, substituir por valor absoluto ou mínimo
                df.loc[df[col] < 0, col] = df[col].abs()
    
    if negative_cols:
        logging.warning(f"[{stage_name}] Encontrados valores negativos (já corrigidos):")
        for col, count in negative_cols:
            logging.warning(f"  - Coluna {col}: {count} valores negativos")
    return df

# Função para verificar valores negativos em um DataFrame
def check_negative_values(df, stage_name):
    negative_cols = []
    for col in df.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_cols.append((col, neg_count))
    
    if negative_cols:
        logging.warning(f"[{stage_name}] Encontrados valores negativos:")
        for col, count in negative_cols:
            logging.warning(f"  - Coluna {col}: {count} valores negativos")
    return df

# Função mantida para compatibilidade (encaminha para a nova função no módulo train_pipeline)
def normalize_data(train_df, val_df, test_df, feature_cols):
    """
    Função de compatibilidade que chama normalize_features do módulo train_pipeline.
    Mantida para não quebrar código existente.
    """
    train_df, val_df, test_df, scaler = normalize_features(train_df, val_df, test_df, feature_cols)
    return train_df, val_df, test_df, scaler

# Carregar configurações dinâmicas
training_cfg = load_config('config/training_config.yaml')
env_cfg = load_config('config/env_config.yaml')
risk_cfg = load_config('config/risk_config.yaml')

# Configuração de performance
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    logging.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("GPU não disponível, usando CPU")

# Carregar múltiplos ativos se necessário
dfs = []
timeframe = env_cfg['environment']['timeframe']
for sym in env_cfg['environment']['symbols']:
    try:
        df = pd.read_csv(f'data/raw/{sym}_{timeframe}.csv', parse_dates=['timestamp'])
        df['symbol'] = sym
        
        # Verificar valores negativos nos dados brutos
        logging.info(f"Verificando dados brutos para {sym}...")
        df = fix_negative_values(df, f"Dados brutos - {sym}")
        
        dfs.append(df)
        logging.info(f"Dados para {sym} carregados com sucesso: {len(df)} linhas.")
    except Exception as e:
        logging.error(f"Erro ao carregar dados para {sym}: {str(e)}")
        
if not dfs:
    logging.error("Nenhum dado foi carregado. Encerrando o script.")
    import sys
    sys.exit(1)
    
df_all = pd.concat(dfs, ignore_index=True)

# Adiciona indicadores técnicos ao DataFrame completo antes da divisão
logging.info("Adicionando indicadores técnicos...")
df_all = add_technical_indicators(df_all)

# Divisão dos dados usando a função prepare_data do módulo train_pipeline
try:
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15  # Calculado automaticamente como 1 - train_ratio - val_ratio

    # Verifica se as proporções são válidas
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or not (0 <= test_ratio < 1):
        raise ValueError("As proporções devem estar entre 0 e 1")
    
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-10:
        raise ValueError(f"A soma das proporções deve ser igual a 1, mas é {train_ratio + val_ratio + test_ratio}")

    logging.info(f"Dividindo dados em treino ({train_ratio*100:.0f}%), "
                f"validação ({val_ratio*100:.0f}%) e teste ({test_ratio*100:.0f}%)")

    # Usa a função prepare_data para fazer a divisão
    train_df, val_df, test_df = prepare_data(
        df=df_all,
        feature_cols=env_cfg['observation']['features'],
        split_ratios=(train_ratio, val_ratio),
        time_col='timestamp',  # Garante a ordem temporal
        shuffle=False  # Importante para manter a ordem temporal
    )
    
    # Verifica a integridade dos dados após a divisão
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Um ou mais conjuntos de dados ficaram vazios após a divisão")
        
    logging.info(f"Divisão concluída: {len(train_df)} treino, {len(val_df)} validação, {len(test_df)} teste")
    
except Exception as e:
    logging.error(f"Erro ao preparar os dados: {str(e)}")
    raise

# Verificar e corrigir valores negativos após adicionar indicadores
logging.info("Verificando e corrigindo valores negativos após adicionar indicadores...")
train_df = fix_negative_values(train_df, "Train após indicadores")
val_df = fix_negative_values(val_df, "Val após indicadores")
test_df = fix_negative_values(test_df, "Test após indicadores")

# Remover NaNs
logging.info("Removendo NaNs...")
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Verificar quantidades após remover NaNs
logging.info(f"Após remover NaNs - Train: {len(train_df)} linhas, Val: {len(val_df)} linhas, Test: {len(test_df)} linhas")

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Aplica RSI a todos os splits
def add_rsi_to_df(df):
    df['rsi'] = calc_rsi(df['close'])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Verificar se algum dos DataFrames ficou vazio
if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    logging.error("Um ou mais DataFrames ficaram vazios após processar NaNs!")
    logging.error(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    import sys
    sys.exit(1)

# Verificação e normalização de features
try:
    feature_cols = env_cfg['observation']['features']
    
    # Verifica se todas as features necessárias estão presentes
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        available_cols = train_df.columns.tolist()
        raise ValueError(
            f"Colunas de features não encontradas: {missing_cols}\n"
            f"Colunas disponíveis: {available_cols}"
        )
    
    # Aplica normalização se necessário
    if env_cfg['observation'].get('normalization', True):
        logging.info(f"Normalizando {len(feature_cols)} features...")
        train_df, val_df, test_df, scaler = normalize_features(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols
        )
        logging.info("Normalização concluída com sucesso")
    else:
        logging.info("Normalização de features desativada na configuração")
        scaler = None
        
except Exception as e:
    logging.error(f"Erro ao processar as features: {str(e)}")
    raise

# Criar diretório para logs com timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("ppo_smc_tensorboard", f"run_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

# Inicializa o gerenciador de risco com configuração personalizada
risk_manager = RiskManager(RiskParameters(
    risk_per_trade=risk_cfg['risk_manager']['risk_per_trade'],
    rr_ratio=risk_cfg['risk_manager']['rr_ratio'],
    max_position_size=0.1,  # Limita a 10% do capital por posição
    max_trades_per_day=1000,  # Valor alto para evitar limitações durante o treinamento
    max_drawdown_percent=0.20,  # Interrompe se drawdown > 20%
    enforce_trade_limit=False  # Desativa o limite de trades para o treinamento
))

# Estratégia com configuração personalizada (ReversalStrategy, TrendFollowingStrategy, ScalpMomentumStrategy, ScalpVWAPStrategy)
strategy = ScalpMomentumStrategy.from_config(risk_cfg['scalp_momentum'])

# Instanciar ambiente, estratégia e risk manager
try:
    logging.info("Criando ambiente de treinamento...")
    env = TradingEnv(
        df=train_df,
        window_size=env_cfg['environment']['window_size'],
        initial_balance=env_cfg['environment']['initial_balance'],
        fee=env_cfg['environment']['fee'],
        symbols=env_cfg['environment']['symbols'],
        strategy=strategy,
        risk_manager=risk_manager
    )
    logging.info("Ambiente de treinamento criado com sucesso!")

    # Ambiente de validação com os mesmos parâmetros
    logging.info("Criando ambiente de validação...")
    eval_env = Monitor(TradingEnv(
        df=val_df,
        window_size=env_cfg['environment']['window_size'],
        initial_balance=env_cfg['environment']['initial_balance'],
        fee=env_cfg['environment']['fee'],
        symbols=env_cfg['environment']['symbols'],
        strategy=strategy,
        risk_manager=risk_manager
    ))
    logging.info("Ambiente de validação criado com sucesso!")
except Exception as e:
    logging.error(f"Erro ao criar ambientes: {str(e)}")
    import sys
    sys.exit(1)

# Instancia modelo RL com parâmetros do YAML e otimizações de performance
class EpsilonGreedyWrapper:
    """Wrapper para adicionar decaimento de epsilon ao modelo PPO."""
    def __init__(self, model, initial_epsilon=1.0, final_epsilon=0.01, decay_steps=1000000):
        self.model = model
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = (initial_epsilon - final_epsilon) / decay_steps

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.decay_rate)

    def predict(self, obs, deterministic=False):
        if np.random.rand() < self.epsilon and not deterministic:
            return self.model.action_space.sample(), None  # Ação aleatória
        else:
            return self.model.predict(obs, deterministic=deterministic)

def softmax_action(model, obs, temperature=1.0):
    """Seleciona ação usando softmax com temperatura ajustável."""
    logits = model.policy.predict(obs)[0]
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return np.random.choice(len(probs), p=probs), None

# Mover a definição do modelo para dentro da função train_agent

def train_agent():
    """
    Função principal para treinar o agente de RL.
    Retorna o modelo treinado e os dados de teste para avaliação.
    """
    try:
        # Cria diretório para logs do TensorBoard
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Nome para o log do TensorBoard
        tb_log_name = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Cria o modelo PPO
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=training_cfg['training']['learning_rate'],
            n_steps=training_cfg['training']['n_steps'],
            gamma=training_cfg['training']['gamma'],
            gae_lambda=training_cfg['training']['gae_lambda'],
            ent_coef=training_cfg['training']['ent_coef'],
            vf_coef=training_cfg['training']['vf_coef'],
            max_grad_norm=training_cfg['training']['max_grad_norm'],
            tensorboard_log=log_dir,
            device=device,
            verbose=1
        )

        # Configura a exploração baseada nas configurações
        if training_cfg['exploration'].get('use_softmax', False):
            exploration_wrapper = lambda obs: softmax_action(model, obs, temperature=training_cfg['exploration'].get('temperature', 0.5))
        else:
            exploration_wrapper = EpsilonGreedyWrapper(
                model,
                initial_epsilon=training_cfg['exploration'].get('initial_epsilon', 1.0),
                final_epsilon=training_cfg['exploration'].get('final_epsilon', 0.01),
                decay_steps=training_cfg['training']['total_timesteps'] // 2
            )

        # Envolve o modelo com o wrapper de epsilon-greedy
        model_wrapped = EpsilonGreedyWrapper(
            model,
            initial_epsilon=1.0,
            final_epsilon=0.01,
            decay_steps=training_cfg['training']['total_timesteps'] // 2  # Decai até metade do treinamento
        )
        # Cria diretório para salvar checkpoints se não existir
        checkpoint_dir = training_cfg['checkpoint']['save_path']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callback para salvar checkpoints periódicos
        checkpoint_callback = CheckpointCallback(
            save_freq=training_cfg['checkpoint']['save_freq'],
            save_path=checkpoint_dir,
            name_prefix='rl_model'
        )
        
        # Callback para avaliação durante o treinamento
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, 'best_model'),
            log_path=log_dir,
            eval_freq=training_cfg['evaluation'].get('eval_freq', 10000),
            deterministic=True,
            render=False,
            n_eval_episodes=training_cfg['evaluation'].get('n_eval_episodes', 5),
            callback_on_new_best=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=training_cfg['evaluation'].get('max_no_improvement', 10),
                min_evals=training_cfg['evaluation'].get('min_evals', 10),
                verbose=1
            ),
            verbose=1
        )
        
        callbacks = [checkpoint_callback, eval_callback]
        
        logging.info(f"Iniciando treinamento por {training_cfg['training']['total_timesteps']} timesteps...")
        
        # Treina o modelo usando a função train_model do módulo train_pipeline
        model = train_model(
            env=env,
            model=model_wrapped.model,  # Usa o modelo interno do wrapper
            total_timesteps=training_cfg['training']['total_timesteps'],
            callbacks=callbacks,
            progress_bar=True,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True
        )
        
        logging.info("Treinamento concluído com sucesso!")
        
        # Retorna o modelo treinado e os dados de teste para avaliação
        return model, test_df
        
    except Exception as e:
        logging.error(f"Erro durante o treinamento do modelo: {str(e)}")
        raise

# Executa o treinamento


def evaluate_model(env, model_wrapped, price_series, plot=True, name='Validação'):
    """
    Avalia o modelo no ambiente fornecido.
    
    Args:
        env: Ambiente de negociação
        model_wrapped: Modelo treinado (já com wrapper de exploração)
        price_series: Série temporal de preços para referência
        plot: Se deve gerar gráficos (não utilizado, mantido para compatibilidade)
        name: Nome do conjunto de dados para logging
    """
    logging.info("\n===================== [EVAL] Início da avaliação =====================")
    obs, _ = env.reset()  # Compatível Gymnasium
    terminated = False
    truncated = False
    trade_log = []
    prev_position = 0
    while not (terminated or truncated):
        if training_cfg['exploration'].get('use_softmax', False):
            action, _ = model_wrapped(obs)  # Usa softmax durante avaliação
        else:
            action, _ = model_wrapped.predict(obs, deterministic=True)  # Usa epsilon-greedy
        obs, _, terminated, truncated, info = env.step(action)
        ts = price_series.index[min(env.unwrapped.current_step, len(price_series)-1)]
        # Marca entradas
        if env.unwrapped.position != prev_position:
            logging.debug(f"[TRADE] [{name}] {'ABERTURA' if env.unwrapped.position != 0 else 'FECHAMENTO'} | Tipo: {env.unwrapped.position} | Preço: {info['entry_price'] if env.unwrapped.position != 0 else price_series.iloc[env.unwrapped.current_step-1]:.2f} | Saldo: {info['balance']:.2f}")
            trade_log.append({
                'timestamp': ts,
                'price': info['entry_price'] if env.unwrapped.position != 0 else price_series.iloc[env.unwrapped.current_step-1],
                'type': 'entry' if env.unwrapped.position != 0 else 'exit',
                'position': env.unwrapped.position,
                'prev_position': prev_position,
                'balance': info['balance'],
                'drawdown': info['max_drawdown']
            })
        # Marca saídas (toda vez que fecha posição)
        if env.unwrapped.position == 0 and prev_position != 0:
            logging.debug(f"[TRADE] [{name}] FECHAMENTO | Preço: {price_series.iloc[env.unwrapped.current_step-1]:.2f} | Saldo: {info['balance']:.2f}")
            trade_log.append({
                'timestamp': ts,
                'price': price_series.iloc[env.unwrapped.current_step-1],
                'type': 'exit',
                'position': env.unwrapped.position,
                'prev_position': prev_position,
                'balance': info['balance'],
                'drawdown': info['max_drawdown']
            })
        prev_position = env.unwrapped.position
    # Métricas simples
    if trade_log:
        final_balance = trade_log[-1]['balance']
        drawdowns = [t['drawdown'] for t in trade_log if 'drawdown' in t]
        n_trades = len([t for t in trade_log if t['type']=='exit'])
        logging.info(f"[EVAL] Saldo final: {final_balance:.2f} | Máx. Drawdown: {min(drawdowns):.2f} | Nº de trades: {n_trades}")
        # if plot:
        #     plot_trades_and_performance(trade_log, price_series)
        # Plot removido, relatório será externo.
    else:
        logging.info(f"[EVAL] Nenhuma operação válida encontrada.")
    logging.info("===================== [EVAL] Fim da avaliação =====================\n")

# Função de plot removida, não é mais utilizada para geração de gráficos ou relatório.
# Nova função de validação com log detalhado para plot

def validate_and_plot(env, model, price_series):
    import pandas as pd
    logging.info("\n===================== [EVAL] Início da validação detalhada =====================")
    obs, _ = env.reset()  # Compatível Gymnasium
    terminated = False
    truncated = False
    trade_log = []
    prev_position = 0
    last_entry_price = None
    last_entry_step = None
    last_entry_obs = None
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs_next, _, terminated, truncated, info = env.step(action)
        ts = price_series.index[min(env.unwrapped.current_step, len(price_series)-1)]
        # Marca entradas
        if env.unwrapped.position != prev_position:
            if env.unwrapped.position != 0:  # Entrada
                last_entry_price = info['entry_price']
                last_entry_step = env.unwrapped.current_step
                last_entry_obs = obs.copy() if hasattr(obs, 'copy') else obs
                trade = {
                    'datetime': ts,
                    'type': 'entry',
                    'preco_entrada': info['entry_price'],
                    'preco_saida': '',
                    'pnl': '',
                    'saldo': info['balance'],
                    'motivo': info.get('event',''),
                    'drawdown': info.get('max_drawdown',''),
                }
                obs_flat = obs.flatten()
                for i in range(len(obs_flat)):
                    trade[f'feature_{i}'] = obs_flat[i]
                trade_log.append(trade)
            else:  # Saída manual
                if last_entry_price is not None:
                    pnl = info['balance'] - trade_log[-1]['saldo']
                    trade = {
                        'datetime': ts,
                        'type': 'exit',
                        'preco_entrada': last_entry_price,
                        'preco_saida': price_series.iloc[env.unwrapped.current_step-1],
                        'pnl': pnl,
                        'saldo': info['balance'],
                        'motivo': info.get('event',''),
                        'drawdown': info.get('max_drawdown',''),
                    }
                    obs_flat = obs.flatten()
                    assert len(obs_flat) == 120, f"Esperado {120} features, mas veio {len(obs_flat)}"
                    for i in range(len(obs_flat)):
                        trade[f'feature_{i}'] = obs_flat[i]
                    trade_log.append(trade)
        # Marca saídas automáticas (stop/target)
        if env.unwrapped.position == 0 and prev_position != 0:
            if last_entry_price is not None:
                pnl = info['balance'] - trade_log[-1]['saldo']
                trade = {
                    'datetime': ts,
                    'type': 'exit',
                    'preco_entrada': last_entry_price,
                    'preco_saida': price_series.iloc[env.unwrapped.current_step-1],
                    'pnl': pnl,
                    'saldo': info['balance'],
                    'motivo': info.get('event',''),
                    'drawdown': info.get('max_drawdown',''),
                }
                obs_last = obs[-1]
                for i in range(obs_last.shape[0]):
                    trade[f'feature_{i}'] = obs_last[i]
                trade_log.append(trade)
        prev_position = env.unwrapped.position
        obs = obs_next
    n_trades = len([t for t in trade_log if t['type']=='exit'])
    logging.info(f"[EVAL] Nº de trades na validação: {n_trades}")
    # plot_trades_and_performance(trade_log, price_series)
    # Gráficos removidos, apenas salva trade_log.csv
    
    # Salva o log de trades para análise posterior
    df_trades = pd.DataFrame(trade_log)
    df_trades.to_csv('trade_log.csv', index=False)
    logging.info(f"Log de trades salvo em trade_log.csv com {len(df_trades)} registros")
    
    return df_trades

# Executa o treinamento e avaliação
if __name__ == "__main__":
    try:
        # Treina o modelo
        trained_model, test_data = train_agent()
        logging.info(f"Modelo treinado com sucesso e salvo em {training_cfg['checkpoint']['save_path']}")
        logging.info(f"Dados de teste disponíveis com {len(test_data)} amostras")

        # Cria o ambiente de teste
        logging.info("Avaliando modelo final no conjunto de teste...")
        test_env = TradingEnv(
            df=test_df,
            window_size=env_cfg['environment']['window_size'],
            initial_balance=env_cfg['environment']['initial_balance'],
            fee=env_cfg['environment']['fee'],
            symbols=env_cfg['environment']['symbols'],
            strategy=strategy,
            risk_manager=risk_manager
        )

        # Avalia o modelo no conjunto de validação
        logging.info("Avaliando modelo no conjunto de validação...")
        val_env = TradingEnv(
            df=val_df,
            window_size=env_cfg['environment']['window_size'],
            initial_balance=env_cfg['environment']['initial_balance'],
            fee=env_cfg['environment']['fee'],
            symbols=env_cfg['environment']['symbols'],
            strategy=strategy,
            risk_manager=risk_manager
        )

        # Gera relatório de performance
        print("\n" + "="*80)
        print("RELATÓRIO DE PERFORMANCE")
        print("="*80)
        print(f"Data da avaliação: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ativos: {', '.join(env_cfg['environment']['symbols'])}")
        print(f"Período: {test_df['timestamp'].min().strftime('%Y-%m-%d')} a {test_df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"Timeframe: {env_cfg['environment']['timeframe']}")
        print("\n" + "-"*80)
        print("ESTRATÉGIA E GESTÃO DE RISCO")
        print("-"*80)
        print(f"Estratégia: {strategy.__class__.__name__ if strategy else 'Nenhuma'}")
        print(f"Gestão de Risco: {risk_manager.__class__.__name__ if risk_manager else 'Nenhuma'}")
        print("\n" + "-"*80)
        print("DESEMPENHO NOS DADOS DE TESTE")
        print("-"*80)

        # Calcula métricas de performance
        initial_balance = env_cfg['environment']['initial_balance']
        final_balance = test_env.balance + test_env.position * test_env.current_price
        returns = (final_balance - initial_balance) / initial_balance * 100
        max_drawdown = test_env.max_drawdown
        n_trades = test_env.n_trades
        win_rate = test_env.win_rate * 100 if hasattr(test_env, 'win_rate') else 0

        print(f"Saldo Inicial: ${initial_balance:,.2f}")
        print(f"Saldo Final: ${final_balance:,.2f}")
        print(f"Retorno: {returns:.2f}%")
        print(f"Máximo Drawdown: {max_drawdown:.2f}%")
        print(f"Número de Trades: {n_trades}")
        print(f"Taxa de Acerto: {win_rate:.2f}%")

        # Valida o modelo no conjunto de teste
        logging.info("Executando validação detalhada no conjunto de teste...")
        trade_log_df = validate_and_plot(test_env, trained_model, test_df['close'])
        
        # Análise de performance
        if not trade_log_df.empty and 'pnl' in trade_log_df.columns:
            # Converte a coluna 'pnl' para tipo numérico (float) antes da comparação
            trade_log_df['pnl'] = pd.to_numeric(trade_log_df['pnl'], errors='coerce')
            
            win_rate = (trade_log_df['pnl'] > 0).mean() * 100
            avg_win = trade_log_df[trade_log_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].mean()
            profit_factor = abs(trade_log_df[trade_log_df['pnl'] > 0]['pnl'].sum() / 
                                trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum()) if trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
            
            print("\n" + "-"*80)
            print("MÉTRICAS DETALHADAS DOS TRADES")
            print("-"*80)
            print(f"- Win Rate: {win_rate:.2f}%")
            print(f"- Profit Factor: {profit_factor:.2f}")
            print(f"- Média de ganhos: ${avg_win:,.2f}" if not pd.isna(avg_win) else "- Média de ganhos: N/A")
            print(f"- Média de perdas: ${avg_loss:,.2f}" if not pd.isna(avg_loss) else "- Média de perdas: N/A")

        # Gera relatório detalhado de trades
        if hasattr(test_env, 'trade_log') and test_env.trade_log:
            print("\n" + "-"*80)
            print("DETALHES DOS TRADES")
            print("-"*80)
            for i, trade in enumerate(test_env.trade_log, 1):
                print(f"\nTrade #{i}")
                print(f"Data: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} a {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"Tipo: {'Compra' if trade['position'] > 0 else 'Venda'}")
                print(f"Preço de Entrada: ${trade['entry_price']:.2f}")
                print(f"Preço de Saída: ${trade['exit_price']:.2f}")
                print(f"Resultado: ${trade['pnl']:+,.2f} ({trade['return']:+.2f}%)")
                print(f"Duração: {trade['duration']} períodos")
                if 'stop_loss' in trade:
                    print(f"Stop Loss: ${trade['stop_loss']:.2f}")
                if 'take_profit' in trade:
                    print(f"Take Profit: ${trade['take_profit']:.2f}")

        print("\n" + "="*80)
        print("FIM DO RELATÓRIO")
        print("="*80)

    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
