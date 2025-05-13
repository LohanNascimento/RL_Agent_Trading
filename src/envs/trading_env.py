import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import logging

from src.utils.risk_manager import RiskManager, RiskParameters

class TradeMetrics:
    """
    Gerencia métricas e histórico de trades com funcionalidades avançadas de análise.
    
    Atributos:
        history (list): Lista de dicionários contendo informações detalhadas de cada trade.
        metrics (dict): Dicionário com métricas agregadas de desempenho.
    """
    def __init__(self):
        """Inicializa o gerenciador de métricas com estruturas de dados vazias."""
        self.history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'recovery_factor': 0.0,
            'avg_trade_duration': 0.0
        }
        self._equity_curve = []
        self._last_equity = 0.0
        self._peak_equity = 0.0
        
    def log_trade(self, trade_data):
        """
        Adiciona uma entrada ao histórico e atualiza as métricas.
        
        Args:
            trade_data (dict): Dicionário contendo informações do trade.
                Deve incluir pelo menos:
                - 'entry_price': Preço de entrada
                - 'position_size': Tamanho da posição
                - 'entry_time': Timestamp de entrada
                - 'type': Tipo de trade ('long' ou 'short')
                - 'status': Status do trade ('open', 'closed', 'sl_hit', 'tp_hit')
                
                Para trades fechados, também deve incluir:
                - 'exit_price': Preço de saída
                - 'exit_time': Timestamp de saída
                - 'pnl': Lucro/Prejuízo do trade
                - 'pnl_pct': Lucro/Prejuízo em porcentagem
                - 'fee': Taxa paga
        """
        try:
            # Valida os dados de entrada
            required_fields = ['entry_price', 'position_size', 'entry_time', 'type', 'status']
            
            # Campos adicionais obrigatórios para trades fechados
            if trade_data.get('status') not in ['open']:
                required_fields.extend(['exit_price', 'exit_time', 'pnl', 'pnl_pct', 'fee'])
            
            for field in required_fields:
                if field not in trade_data:
                    raise ValueError(f"Campo obrigatório não encontrado: {field}")
            
            # Adiciona o trade ao histórico
            self.history.append(trade_data)
            
            # Atualiza as métricas apenas para trades fechados
            if trade_data.get('status') not in ['open']:
                self._update_metrics(trade_data)
                
                logging.info(f"Trade registrado: {trade_data['type'].upper()} | "
                            f"Entrada: {trade_data['entry_price']:.2f} | "
                            f"Saída: {trade_data['exit_price']:.2f} | "
                            f"P&L: {trade_data['pnl']:.2f} ({trade_data['pnl_pct']:.2f}%)")
            else:
                # Log para trades abertos
                logging.info(f"Trade aberto: {trade_data['type'].upper()} | "
                            f"Entrada: {trade_data['entry_price']:.2f} | "
                            f"Tamanho: {trade_data['position_size']:.4f}")
                        
        except Exception as e:
            logging.error(f"Erro ao registrar trade: {str(e)}")
            raise
    
    def _update_metrics(self, trade_data):
        """
        Atualiza as métricas com base no trade mais recente.
        
        Args:
            trade_data (dict): Dados do trade mais recente
        """
        try:
            # Atualiza contadores básicos
            self.metrics['total_trades'] = len(self.history)
            
            # Atualiza contagem de trades vencedores e perdedores
            pnl = trade_data.get('pnl', 0)
            if pnl > 0:
                self.metrics['winning_trades'] += 1
                self.metrics['total_profit'] += pnl
                self.metrics['max_profit'] = max(self.metrics['max_profit'], pnl)
                
                # Atualiza média de lucro
                if self.metrics['winning_trades'] > 0:
                    self.metrics['avg_profit'] = (
                        (self.metrics['avg_profit'] * (self.metrics['winning_trades'] - 1) + pnl) / 
                        self.metrics['winning_trades']
                    )
            else:
                self.metrics['losing_trades'] += 1
                self.metrics['total_loss'] += abs(pnl)
                self.metrics['max_loss'] = min(self.metrics['max_loss'], pnl)
                
                # Atualiza média de perda
                if self.metrics['losing_trades'] > 0:
                    self.metrics['avg_loss'] = (
                        (self.metrics['avg_loss'] * (self.metrics['losing_trades'] - 1) + abs(pnl)) / 
                        self.metrics['losing_trades']
                    )
            
            # Atualiza métricas derivadas
            if self.metrics['total_trades'] > 0:
                self.metrics['win_rate'] = (
                    self.metrics['winning_trades'] / self.metrics['total_trades'] * 100
                )
                
                if self.metrics['total_loss'] != 0:
                    self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
                
                # Calcula o drawdown
                self._last_equity += pnl
                self._peak_equity = max(self._peak_equity, self._last_equity)
                drawdown = self._peak_equity - self._last_equity
                self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
                
                # Atualiza a curva de equity
                self._equity_curve.append(self._last_equity)
                
                # Calcula métricas de risco-retorno
                self._calculate_risk_metrics()
                
        except Exception as e:
            logging.error(f"Erro ao atualizar métricas: {str(e)}")
    
    def _calculate_risk_metrics(self):
        """Calcula métricas de risco-retorno baseadas na curva de equity."""
        if len(self._equity_curve) < 2:
            return
            
        try:
            returns = np.diff(self._equity_curve) / np.array(self._equity_curve[:-1])
            
            # Sharpe Ratio (assumindo taxa livre de risco = 0)
            if np.std(returns) > 0:
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
            
            # Sortino Ratio (considera apenas desvios negativos)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                self.metrics['sortino_ratio'] = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
            
            # Recovery Factor (lucro total / drawdown máximo)
            if self.metrics['max_drawdown'] > 0:
                self.metrics['recovery_factor'] = self._last_equity / self.metrics['max_drawdown']
                
        except Exception as e:
            logging.error(f"Erro ao calcular métricas de risco: {str(e)}")
    
    def get_history(self):
        """
        Retorna o histórico de trades como um DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame contendo o histórico de trades com colunas detalhadas.
        """
        if not self.history:
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(self.history)
            
            # Adiciona colunas calculadas, se necessário
            if 'entry_time' in df.columns and 'exit_time' in df.columns:
                df['duration'] = df['exit_time'] - df['entry_time']
                
            return df
            
        except Exception as e:
            logging.error(f"Erro ao converter histórico para DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def get_summary(self):
        """
        Retorna um resumo das métricas de desempenho.
        
        Returns:
            dict: Dicionário com métricas de desempenho.
        """
        return self.metrics.copy()
        
    def get_equity_curve(self):
        """
        Retorna a curva de equity ao longo do tempo.
        
        Returns:
            list: Lista com o valor da equity em cada ponto no tempo.
        """
        return self._equity_curve.copy()
    
    def reset(self):
        """Reinicia todas as métricas e histórico."""
        self.history = []
        self._equity_curve = []
        self._last_equity = 0.0
        self._peak_equity = 0.0
        
        # Reinicializa as métricas
        for key in self.metrics:
            if isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0.0
            elif isinstance(self.metrics[key], (list, dict)):
                self.metrics[key] = type(self.metrics[key])()

class TradingEnv(gym.Env):
    """
    Ambiente de RL customizado para trading com Smart Money Concepts (SMC).
    O vetor de estado inclui preço, volume e indicadores técnicos.
    Ações discretas: 0 = Manter, 1 = Comprar, 2 = Vender.
    Recompensa baseada em P&L líquido e penalização de risco.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=10, initial_balance=1000, fee=0.001, strategy=None, risk_manager=None, symbols=None, log_level=logging.INFO, enforce_max_drawdown=True):
        super().__init__()
        # Configura logging
        logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s:%(message)s')
        
        self.strategy = strategy  # Garante que sempre existe o atributo strategy
        self.symbols = symbols
        self.df = df.reset_index(drop=True)
        self.window_size = window_size  # Definido antes da validação
        self.enforce_max_drawdown = enforce_max_drawdown  # Flag para ativar/desativar controle de drawdown
        
        # Validação dos dados de entrada
        self._validate_dataframe()
        
        self.initial_balance = initial_balance
        self.fee = fee
        self.stop_loss = None
        self.take_profit = None
        
        # Define espaços de ação e observação
        self.action_space = spaces.Discrete(3)
        
        # Lista de indicadores técnicos
        self.indicator_cols = [
            'sma_14', 'ema_21', 'macd', 'macd_signal', 'std_14', 'atr_14',
            'bb_upper', 'bb_lower', 'rsi_14', 'stoch_k', 'stoch_d', 'roc_14'
        ]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(self.indicator_cols)), dtype=np.float32
        )
        
        # Inicializa o RiskManager com parâmetros padrão ou customizados
        if isinstance(risk_manager, RiskManager):
            self.risk_manager = risk_manager
        elif isinstance(risk_manager, dict):
            self.risk_manager = RiskManager(RiskParameters(**risk_manager))
        else:
            # Parâmetros padrão
            self.risk_manager = RiskManager()
        
        # Inicializa o gerenciador de métricas
        self.metrics = TradeMetrics()
        
        self.risk_manager.log_risk_parameters()
        self.reset()

    def _validate_dataframe(self):
        """Valida o DataFrame para garantir integridade dos dados"""
        # Verifica se temos as colunas necessárias
        required_cols = ['close', 'open', 'high', 'low']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame deve conter a coluna {col}")
                
        # Verifica valores inválidos (NaN, infinitos, negativos)
        for col in required_cols:
            if self.df[col].isna().any():
                logging.warning(f"Encontradas {self.df[col].isna().sum()} linhas com NaN na coluna {col}. Removendo...")
                self.df = self.df.dropna(subset=[col])
                
            if (self.df[col] < 0).any():
                logging.warning(f"Encontradas {(self.df[col] < 0).sum()} linhas com valores negativos na coluna {col}. Removendo...")
                self.df = self.df[self.df[col] > 0]  # Alterado: remove apenas valores negativos, mantém zeros
                
        # Ordena pelo timestamp, se existir
        if 'timestamp' in self.df.columns:
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
        if len(self.df) <= self.window_size:
            raise ValueError(f"DataFrame muito pequeno: {len(self.df)} linhas. Necessário pelo menos {self.window_size + 1} linhas.")

    def _get_observation(self):
        """
        Retorna a observação atual do ambiente.
        
        Returns:
            np.ndarray: Array de formato (window_size, n_indicators) contendo os valores
                      dos indicadores técnicos para a janela atual.
        """
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        obs = []
        
        # Verifica quais indicadores estão disponíveis no DataFrame
        available_indicators = [col for col in self.indicator_cols if col in window.columns]
        missing_indicators = set(self.indicator_cols) - set(available_indicators)
        
        if missing_indicators:
            logging.warning(f"Indicadores não encontrados no DataFrame: {missing_indicators}")
            logging.warning("Substituindo por zeros. Verifique se os indicadores foram calculados corretamente.")
        
        for i in range(len(window)):
            row = window.iloc[i]
            obs_row = []
            for col in self.indicator_cols:
                if col in available_indicators and not pd.isna(row[col]):
                    # Converte para float32 diretamente para evitar problemas de tipo
                    obs_row.append(float(row[col]))
                else:
                    # Preenche com zero se o indicador estiver ausente ou for NaN
                    obs_row.append(0.0)
            obs.append(obs_row)
        
        # Converte para array numpy e garante o tipo float32
        obs = np.array(obs, dtype=np.float32)
        
        # Substitui quaisquer valores NaN, infinitos ou muito grandes por zero
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verificação final de integridade
        if np.isnan(obs).any() or np.isinf(obs).any():
            logging.error(f"Valores inválidos encontrados na observação: {obs}")
            # Se ainda houver valores inválidos, substitui por zero
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o estado inicial.
        
        Args:
            seed: Semente para reprodutibilidade
            options: Opções adicionais para o reset
            
        Returns:
            obs: Observação inicial
            info: Dicionário com informações adicionais
        """
        # Reseta o estado do ambiente
        self.balance = self.initial_balance
        self.position = 0  # +1 comprado, -1 vendido, 0 neutro
        self.entry_price = 0
        self.current_price = 0  # Inicializa o preço atual
        self.current_step = self.window_size
        self.done = False
        self.total_profit = 0
        self.max_drawdown = 0
        self.episode_profit = 0
        self.last_balance = self.initial_balance
        self.trade_count = 0
        self.n_trades = 0  # Adiciona contador de trades acessível externamente
        
        # Reseta o histórico de preços no risk manager
        self.risk_manager.price_history = []
        
        # Reseta as métricas
        self.metrics = TradeMetrics()
        
        # Obtém a observação inicial
        obs = self._get_observation()
        assert not np.isnan(obs).any(), f"Encontrado NaN na observação no reset: {obs}"
        
        return obs, {}

    def step(self, action):
        """
        Executa um passo no ambiente de negociação.
        
        Args:
            action: Ação a ser executada (0 = Manter, 1 = Comprar, 2 = Vender)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # Valida a ação
            if not self.action_space.contains(action):
                raise ValueError(f"Ação inválida: {action}. Deve ser 0, 1 ou 2.")
                
            # Obtém a linha atual do DataFrame
            if self.current_step >= len(self.df):
                raise IndexError(f"Índice {self.current_step} fora dos limites do DataFrame com tamanho {len(self.df)}")
                
            row = self.df.iloc[self.current_step]

            # Adicionar penalidade proporcional ao drawdown se estiver ativado
            if self.enforce_max_drawdown and self.max_drawdown > 0.1:  # 10% de drawdown
                reward -= self.max_drawdown * 10  # Penalidade proporcional

            # Obtém o preço de fechamento atual
            if 'close' not in row:
                raise KeyError("Coluna 'close' não encontrada no DataFrame")
                
            price = float(row["close"])
            self.current_price = price  # Atualiza o preço atual
            reward = 0.0
            info = {
                'step': self.current_step,
                'price': price,
                'balance': self.balance,
                'position': self.position,
                'event': None,
                'error': None
            }
            terminated = False
            truncated = False
            event = None
            
            # Valida o preço
            if np.isnan(price) or np.isinf(price) or price <= 0:
                error_msg = f"Preço inválido detectado em step {self.current_step}: {price}"
                logging.error(error_msg)
                info['error'] = error_msg
                terminated = True
                return self._get_observation(), -100, terminated, truncated, info
                
            # Verifica limites de risco antes de qualquer ação
            try:
                if self.enforce_max_drawdown:
                    risk_check = self.risk_manager.check_trade_limits(self.balance, price)
                    if not risk_check['allowed'] and action != 0:  # Só permite manter posição
                        warning_msg = f"Ação bloqueada: {risk_check['reason']}"
                        logging.warning(warning_msg)
                        event = f"blocked_{risk_check['reason'].replace(' ', '_')}"
                        action = 0  # Força "manter"
                        # Pequena penalidade por tentar violar limites de risco
                        reward = -0.01
                        info['event'] = event
                        info['warning'] = warning_msg
                else:
                    # Quando o controle de drawdown está desativado, apenas registra o drawdown sem bloquear ações
                    risk_check = self.risk_manager.check_trade_limits(self.balance, price, check_only=True)
                    if not risk_check['allowed']:
                        logging.info(f"Drawdown detectado ({risk_check['reason']}), mas ignorado devido a enforce_max_drawdown=False")
                        info['drawdown_detected'] = risk_check['reason']
            except Exception as e:
                error_msg = f"Erro ao verificar limites de risco: {str(e)}"
                logging.error(error_msg)
                info['error'] = error_msg
                terminated = True
                return self._get_observation(), -100, terminated, truncated, info
                
            # Calcula stop loss e take profit antes para determinar alavancagem
            try:
                if action in [1, 2] and self.position == 0:
                    entry_type = 'long' if action == 1 else 'short'
                    temp_stop_loss, _ = self._calculate_risk_levels(price, entry_type)
                    
                    # Calcula distância percentual até o stop loss
                    stop_distance = abs(price - temp_stop_loss) / price
                    
                    # Calcula alavancagem baseada na distância do stop loss
                    # Quanto maior a distância, menor a alavancagem para manter risco constante
                    leverage = min(20, 0.02 / stop_distance)  # Máximo de 20x, risco alvo de 2%
                    
                    # Ajusta o tamanho da posição com base na alavancagem
                    position_size = self.risk_manager.calculate_position_size(self.balance, price) * leverage
                else:
                    position_size = self.risk_manager.calculate_position_size(self.balance, price)
                    
                # Limita o tamanho da posição para evitar valores extremos
                max_safe_position = 1000.0 / price  # Limita o valor máximo da posição
                position_size = min(position_size, max_safe_position)
                
            except Exception as e:
                error_msg = f"Erro ao calcular tamanho da posição: {str(e)}"
                logging.error(error_msg)
                info['error'] = error_msg
                terminated = True
                return self._get_observation(), -100, terminated, truncated, info
                
            # Execução da ação
            try:
                if action == 1 and self.position == 0:  # Comprar
                    self.position = 1
                    self.entry_price = price
                    self.entry_step = self.current_step
                    # Calcula stop loss e take profit para long
                    self.stop_loss, self.take_profit = self._calculate_risk_levels(price, 'long')
                    
                    # Registra a abertura da posição longa
                    trade_data = self._create_trade_data(
                        entry_price=price,
                        exit_price=price,  # Mesmo preço para abertura
                        position_size=position_size,
                        trade_type='short',
                        status='open',
                        step=self.current_step,
                        entry_step=self.current_step
                    )
                    trade_data.update({
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                    self.metrics.log_trade(trade_data)
                    
                    logging.info(f"[ABERTURA] LONG em {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f} | Saldo: {self.balance:.2f}")
                    event = 'open_long'
                    self.trade_count += 1
                    self.n_trades += 1  # Incrementa o contador de trades
                    
                    self.risk_manager.register_trade({
                        'type': 'open_long',
                        'price': price,
                        'position_size': position_size,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                    info['event'] = event
                    info['trade_data'] = trade_data
                    
                elif action == 2 and self.position == 0:  # Vender
                    self.position = -1
                    self.entry_price = price
                    self.entry_step = self.current_step
                    # Calcula stop loss e take profit para short
                    self.stop_loss, self.take_profit = self._calculate_risk_levels(price, 'short')
                    
                    # Registra a abertura da posição curta
                    trade_data = self._create_trade_data(
                        entry_price=price,
                        exit_price=price,  # Mesmo preço para abertura
                        position_size=position_size,
                        trade_type='short',
                        status='open',
                        step=self.current_step,
                        entry_step=self.current_step  # Adicionado entry_step
                    )
                    trade_data.update({
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                    self.metrics.log_trade(trade_data)
                    
                    logging.info(f"[ABERTURA] SHORT em {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f} | Saldo: {self.balance:.2f}")
                    event = 'open_short'
                    self.trade_count += 1
                    self.n_trades += 1  # Incrementa o contador de trades
                    
                    self.risk_manager.register_trade({
                        'type': 'open_short',
                        'price': price,
                        'position_size': position_size,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                    info['event'] = event
                    info['trade_data'] = trade_data
                    
                elif action == 2 and self.position == 1:  # Fecha compra manualmente
                    try:
                        # Calcula o lucro/prejuízo
                        profit = (price - self.entry_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((price / self.entry_price) - 1) * 100 if self.entry_price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=price,
                            position_size=position_size,
                            trade_type='short',
                            status='closed_manual',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[FECHAMENTO MANUAL] LONG em {price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'close_long_manual'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'close_long_manual',
                            'price': price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                        
                    except Exception as e:
                        error_msg = f"Erro ao fechar posição longa: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        info['error'] = error_msg
                        
                elif action == 1 and self.position == -1:  # Fecha venda manualmente
                    try:
                        # Calcula o lucro/prejuízo
                        profit = (self.entry_price - price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((self.entry_price / price) - 1) * 100 if price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=price,
                            position_size=position_size,
                            trade_type='short',
                            status='closed_manual',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[FECHAMENTO MANUAL] SHORT em {price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'close_short_manual'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'close_short_manual',
                            'price': price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                        
                    except Exception as e:
                        error_msg = f"Erro ao fechar posição curta: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        info['error'] = error_msg
                        
                # Atualiza o passo atual
                self.current_step += 1
                
                # Verifica se o episódio terminou
                if self.current_step >= len(self.df) - 1:
                    terminated = True
                    
                # Atualiza o saldo no info
                info['balance'] = self.balance
                info['position'] = self.position
                
                # Obtém a próxima observação
                obs = self._get_observation()
                
                return obs, reward, terminated, truncated, info
                
            except Exception as e:
                error_msg = f"Erro ao executar ação: {str(e)}"
                logging.error(error_msg)
                info['error'] = error_msg
                terminated = True
                return self._get_observation(), -100, terminated, truncated, info
                
        except Exception as e:
            # Captura qualquer exceção não tratada
                error_msg = f"Erro inesperado em step: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return self._get_observation(), -100, True, False, {'error': error_msg}

        # Checagem automática de stop loss/take profit
        if self.position != 0 and hasattr(self, 'stop_loss') and hasattr(self, 'take_profit'):
            if self._hit_stop_or_target(price):
                if self.position == 1:  # LONG
                    if price <= self.stop_loss:
                        # Stop Loss LONG
                        exit_price = min(self.stop_loss, price)  # Usa o menor preço entre stop e preço atual
                        profit = (exit_price - self.entry_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((exit_price / self.entry_price) - 1) * 100 if self.entry_price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=exit_price,
                            position_size=position_size,
                            trade_type='short',
                            status='stop_loss',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[STOP LOSS] LONG atingido em {exit_price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'stop_loss_long'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'stop_loss_long',
                            'price': exit_price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                    elif price >= self.take_profit:
                        # Take Profit LONG
                        exit_price = max(self.take_profit, price)  # Usa o maior preço entre take profit e preço atual
                        profit = (exit_price - self.entry_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((exit_price / self.entry_price) - 1) * 100 if self.entry_price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=exit_price,
                            position_size=position_size,
                            trade_type='short',
                            status='take_profit',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[TAKE PROFIT] LONG atingido em {exit_price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'take_profit_long'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'take_profit_long',
                            'price': exit_price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                elif self.position == -1:  # SHORT
                    if price >= self.stop_loss:
                        # Stop Loss SHORT
                        exit_price = max(self.stop_loss, price)  # Usa o maior preço entre stop e preço atual
                        profit = (self.entry_price - exit_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((self.entry_price / exit_price) - 1) * 100 if exit_price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=exit_price,
                            position_size=position_size,
                            trade_type='short',
                            status='stop_loss',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[STOP LOSS] SHORT atingido em {exit_price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'stop_loss_short'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'stop_loss_short',
                            'price': exit_price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                    elif price <= self.take_profit:
                        # Take Profit SHORT
                        exit_price = min(self.take_profit, price)  # Usa o menor preço entre take profit e preço atual
                        profit = (self.entry_price - exit_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        pnl_pct = ((self.entry_price / exit_price) - 1) * 100 if exit_price != 0 else 0
                        
                        # Atualiza o saldo e a posição
                        self.balance += profit
                        self.position = 0
                        
                        # Registra o fechamento da posição
                        trade_data = self._create_trade_data(
                            entry_price=self.entry_price,
                            exit_price=exit_price,
                            position_size=position_size,
                            trade_type='short',
                            status='take_profit',
                            step=self.current_step,
                            entry_step=self.entry_step if hasattr(self, 'entry_step') else self.current_step
                        )
                        self.metrics.log_trade(trade_data)
                        
                        reward = profit
                        logging.info(f"[TAKE PROFIT] SHORT atingido em {exit_price:.2f} | PnL: {profit:.2f} ({pnl_pct:.2f}%) | Saldo: {self.balance:.2f}")
                        event = 'take_profit_short'
                        self.n_trades += 1  # Incrementa o contador de trades
                        
                        self.risk_manager.register_trade({
                            'type': 'take_profit_short',
                            'price': exit_price,
                            'profit': profit,
                            'pnl_pct': pnl_pct,
                            'duration': trade_data['duration']
                        })
                        
                        # Reseta os parâmetros da posição
                        self.entry_price = 0
                        self.entry_step = None
                        self.stop_loss = None
                        self.take_profit = None
                        
                        info['event'] = event
                        info['trade_data'] = trade_data
                self.balance += reward
                # Limita o valor do saldo para evitar overflow
                self.balance = np.clip(self.balance, 0, 1e9)
                self.position = 0
                self.entry_price = 0

        # Atualiza métricas de performance
        self.episode_profit = self.balance - self.initial_balance
        drawdown = min(0, self.episode_profit)
        self.max_drawdown = min(self.max_drawdown, drawdown)

        # Métricas de risco
        sharpe_ratio = self.episode_profit / (np.std([self.balance, self.initial_balance]) + 1e-8)
        calmar_ratio = self.episode_profit / (abs(self.max_drawdown) + 1e-8) if self.max_drawdown < 0 else 0
        
        # Penalidade por drawdown severo (adiciona pressão para controle de risco)
        if self.max_drawdown < -self.initial_balance * 0.1:  # Drawdown > 10%
            reward = reward * 0.8  # Reduz recompensa

        # Adicionar penalidade proporcional ao drawdown se estiver ativado
        if self.enforce_max_drawdown and self.max_drawdown > 0.1:  # 10% de drawdown
            reward -= self.max_drawdown * 10  # Penalidade proporcional

        # Atualiza estado
        self.current_step += 1
        if self.current_step >= len(self.df) or self.balance <= 0:
            self.done = True
            terminated = True

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "episode_profit": self.episode_profit,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "event": event,
            "position_size": position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_per_trade": self.risk_manager.params.risk_per_trade,
            "rr_ratio": self.risk_manager.params.rr_ratio,
            "trade_count": self.trade_count
        }
        return obs, reward, terminated, truncated, info

    def _calculate_risk_levels(self, price, entry_type=None):
        """
        Calcula stop loss e take profit para uma entrada baseada no preço atual e tipo de entrada.
        
        Args:
            price: Preço de entrada
            entry_type: 'long' ou 'short'
            
        Returns:
            Tuple (stop_loss, take_profit)
            
        Raises:
            ValueError: Se o preço for inválido ou o tipo de entrada for desconhecido
        """
        try:
            # Valida o preço
            if np.isnan(price) or np.isinf(price) or price <= 0:
                raise ValueError(f"Preço inválido: {price}")
                
            # Determina o tipo de posição se não for fornecido
            if entry_type is None and hasattr(self, 'position'):
                entry_type = 'long' if self.position == 1 else 'short'
            elif entry_type is None:
                entry_type = 'long'  # Valor padrão
                
            # Valida o tipo de entrada
            if entry_type not in ['long', 'short']:
                raise ValueError(f"Tipo de entrada inválido: {entry_type}. Deve ser 'long' ou 'short'.")
                
            # Verifica se o gerenciador de risco está disponível
            if not hasattr(self, 'risk_manager') or not hasattr(self.risk_manager, 'params'):
                logging.warning("Risk manager não disponível, usando valores padrão")
                risk_params = type('Params', (), {
                    'rr_ratio': 2.0,
                    'risk_per_trade': 0.02
                })()
            else:
                risk_params = self.risk_manager.params
            
            stop_loss = 0.0
            take_profit = 0.0
                
            # Garante que self.strategy existe e é do tipo esperado
            is_trend_following = (
                hasattr(self, 'strategy') and 
                self.strategy is not None and 
                getattr(self.strategy, '__class__', None) and 
                self.strategy.__class__.__name__ == 'TrendFollowingStrategy'
            )
            
            if is_trend_following:
                if entry_type == 'long':  # Long
                    # Para long, stop loss abaixo do preço
                    try:
                        stop_loss = self._find_previous_ll(price)  # Lower Low para long
                        # Limita o stop loss para valores razoáveis (máx. 30% abaixo)
                        stop_loss = max(stop_loss, price * 0.7)
                        
                        # Calcula take profit com base no stop loss
                        tp_distance = (price - stop_loss) * risk_params.rr_ratio
                        take_profit = price + tp_distance
                        # Limita o take profit para valores razoáveis (máx. 100% acima)
                        take_profit = min(take_profit, price * 2.0)
                        
                        logging.debug(f"[LONG] Preço: {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                        
                    except Exception as e:
                        logging.error(f"Erro ao calcular níveis de risco para LONG: {str(e)}")
                        # Usa valores padrão em caso de erro
                        stop_loss = price * 0.98
                        take_profit = price * (1 + 0.02 * risk_params.rr_ratio)
                        
                else:  # Short
                    try:
                        # Para short, stop loss acima do preço
                        stop_loss = self._find_previous_hh(price)  # Higher High para short
                        # Limita o stop loss para valores razoáveis (máx. 30% acima)
                        stop_loss = min(stop_loss, price * 1.3)
                        
                        # Calcula take profit com base no stop loss
                        tp_distance = (stop_loss - price) * risk_params.rr_ratio
                        take_profit = price - tp_distance
                        # Limita o take profit para valores razoáveis (mín. 50% do preço)
                        take_profit = max(take_profit, price * 0.5)
                        
                        logging.debug(f"[SHORT] Preço: {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                        
                    except Exception as e:
                        logging.error(f"Erro ao calcular níveis de risco para SHORT: {str(e)}")
                        # Usa valores padrão em caso de erro
                        stop_loss = price * 1.02
                        take_profit = price * (1 - 0.02 * risk_params.rr_ratio)
                        
            else:
                # Lógica padrão para outras estratégias
                if entry_type == 'long':  # Long
                    stop_loss = price * 0.98  # Stop loss padrão 2% abaixo
                    take_profit = price * (1 + risk_params.risk_per_trade * risk_params.rr_ratio)
                    # Limita o take profit para valores razoáveis (máx. 10% acima)
                    take_profit = min(take_profit, price * 1.1)
                    
                    logging.debug(f"[LONG PADRAO] Preço: {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                    
                else:  # Short
                    stop_loss = price * 1.02  # Stop loss padrão 2% acima
                    take_profit = price * (1 - risk_params.risk_per_trade * risk_params.rr_ratio)
                    # Limita o take profit para valores razoáveis (mín. 10% abaixo)
                    take_profit = max(take_profit, price * 0.9)
                    
                    logging.debug(f"[SHORT PADRAO] Preço: {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            
            # Garante que os valores são válidos
            if np.isnan(stop_loss) or np.isinf(stop_loss) or stop_loss <= 0:
                logging.warning(f"Valor de stop loss inválido: {stop_loss}, usando valor padrão")
                stop_loss = price * (0.98 if entry_type == 'long' else 1.02)
                
            if np.isnan(take_profit) or np.isinf(take_profit) or take_profit <= 0:
                logging.warning(f"Valor de take profit inválido: {take_profit}, usando valor padrão")
                risk_multiplier = 1 + (0.02 * risk_params.rr_ratio) if entry_type == 'long' else 1 - (0.02 * risk_params.rr_ratio)
                take_profit = price * risk_multiplier
                
            # Garante que o stop loss está do lado correto do preço
            if entry_type == 'long' and stop_loss >= price:
                logging.warning(f"Stop loss ({stop_loss}) maior ou igual ao preço ({price}) para LONG, ajustando...")
                stop_loss = price * 0.99
            elif entry_type == 'short' and stop_loss <= price:
                logging.warning(f"Stop loss ({stop_loss}) menor ou igual ao preço ({price}) para SHORT, ajustando...")
                stop_loss = price * 1.01
                
            # Garante que o take profit está do lado correto do preço
            if entry_type == 'long' and take_profit <= price:
                logging.warning(f"Take profit ({take_profit}) menor ou igual ao preço ({price}) para LONG, ajustando...")
                take_profit = price * 1.01
            elif entry_type == 'short' and take_profit >= price:
                logging.warning(f"Take profit ({take_profit}) maior ou igual ao preço ({price}) para SHORT, ajustando...")
                take_profit = price * 0.99
            
            return stop_loss, take_profit
            
        except Exception as e:
            logging.error(f"Erro crítico em _calculate_risk_levels: {str(e)}", exc_info=True)
            # Retorna valores padrão em caso de erro inesperado
            default_ratio = 0.02
            if entry_type == 'long':
                return price * (1 - default_ratio), price * (1 + default_ratio * 2)
            else:
                return price * (1 + default_ratio), price * (1 - default_ratio * 2)

    def _create_trade_data(self, entry_price, exit_price, position_size, trade_type, status, step, entry_step=None):
        """
        Cria um dicionário padronizado com os dados do trade.
        
        Args:
            entry_price: Preço de entrada
            exit_price: Preço de saída
            position_size: Tamanho da posição
            trade_type: Tipo de trade ('long' ou 'short')
            status: Status do trade ('open', 'closed', 'stop_loss', 'take_profit', 'closed_manual')
            step: Passo atual
            entry_step: Passo de entrada (opcional)
            
        Returns:
            dict: Dicionário com os dados do trade
        """
        timestamp = self.df.index[step] if hasattr(self.df.index, 'name') and self.df.index.name == 'timestamp' else step
        fee = exit_price * position_size * self.fee if status != 'open' else entry_price * position_size * self.fee
        
        if status == 'open':
            return {
                'entry_time': timestamp,
                'entry_price': entry_price,
                'position_size': position_size,
                'type': trade_type,
                'status': status,
                'fee': fee,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'step': step
            }
        else:
            pnl = (exit_price - entry_price) * position_size * (1 - self.fee) if trade_type == 'long' else \
                  (entry_price - exit_price) * position_size * (1 - self.fee)
            pnl_pct = ((exit_price / entry_price) - 1) * 100 if trade_type == 'long' else \
                     ((entry_price / exit_price) - 1) * 100
            duration = step - entry_step if entry_step is not None else 0
            
            return {
                'entry_time': self.df.index[entry_step] if hasattr(self.df.index, 'name') and self.df.index.name == 'timestamp' else entry_step,
                'exit_time': timestamp,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'type': trade_type,
                'status': status,
                'fee': fee,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration': duration,
                'step': step
            }
    
    def _hit_stop_or_target(self, price):
        """
        Verifica se o preço atual atingiu o stop loss ou take profit.
        
        Args:
            price: Preço atual do ativo
            
        Returns:
            bool: True se o preço atingiu stop loss ou take profit, False caso contrário
        """
        try:
            # Valida os parâmetros de entrada
            if not hasattr(self, 'position') or self.position == 0:
                return False
                
            if not hasattr(self, 'stop_loss') or not hasattr(self, 'take_profit'):
                logging.warning("Stop loss ou take profit não definidos")
                return False
                
            if np.isnan(price) or np.isinf(price) or price <= 0:
                logging.warning(f"Preço inválido na verificação de stop/target: {price}")
                return False
                
            # Verifica se o preço atingiu o stop loss ou take profit
            hit = False
            
            if self.position == 1:  # Posição longa
                if price <= self.stop_loss:
                    logging.info(f"[STOP LOSS] LONG atingido em {price:.2f} (SL: {self.stop_loss:.2f})")
                    hit = True
                elif price >= self.take_profit:
                    logging.info(f"[TAKE PROFIT] LONG atingido em {price:.2f} (TP: {self.take_profit:.2f})")
                    hit = True
                    
            elif self.position == -1:  # Posição curta
                if price >= self.stop_loss:
                    logging.info(f"[STOP LOSS] SHORT atingido em {price:.2f} (SL: {self.stop_loss:.2f})")
                    hit = True
                elif price <= self.take_profit:
                    logging.info(f"[TAKE PROFIT] SHORT atingido em {price:.2f} (TP: {self.take_profit:.2f})")
                    hit = True
            
            return hit
            
        except Exception as e:
            logging.error(f"Erro ao verificar stop/target: {str(e)}")
            return False
        
    def render(self, mode="human"):
        if mode == "human":
            logging.info(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position}")
        else:
            super().render(mode=mode)

    def _find_previous_hh(self, price):
        """
        Encontra o Higher High (HH) anterior para usar como stop loss.
        
        Args:
            price: Preço atual do ativo
            
        Returns:
            float: O maior preço (high) no período de lookback, ajustado se necessário
            
        Raises:
            ValueError: Se o preço for inválido ou não for possível determinar o HH
        """
        try:
            # Valida o preço
            if np.isnan(price) or np.isinf(price) or price <= 0:
                raise ValueError(f"Preço inválido: {price}")
                
            # Define o tamanho da janela de lookback
            lookback = min(20, self.current_step)
            if lookback <= 0:
                logging.warning("Lookback muito pequeno, usando valor padrão de 10")
                lookback = 10
                
            # Obtém a janela de dados
            start_idx = max(0, self.current_step - lookback)
            window = self.df.iloc[start_idx:self.current_step]
            
            # Verifica se a coluna 'high' existe
            if 'high' not in window.columns:
                logging.warning("Coluna 'high' não encontrada, usando 2% acima do preço")
                return price * 1.02
                
            # Encontra o maior high no período
            highest_high = window['high'].max()
            
            # Valida o valor encontrado
            if np.isnan(highest_high) or np.isinf(highest_high) or highest_high <= 0:
                logging.warning(f"Valor de highest_high inválido: {highest_high}, usando 2% acima do preço")
                return price * 1.02
                
            # Se o preço atual estiver muito próximo do maior high, 
            # usamos um valor padrão de stop loss (2% abaixo)
            if abs(highest_high - price) / price < 0.01:  # menos de 1% de distância
                logging.debug(f"Preço muito próximo do maior high ({highest_high:.2f}), usando 2% abaixo do preço")
                return price * 0.98
                
            logging.debug(f"Higher High encontrado: {highest_high:.2f} (preço atual: {price:.2f})")
            return highest_high
            
        except Exception as e:
            logging.error(f"Erro ao encontrar Higher High: {str(e)}")
            # Retorna um valor padrão seguro em caso de erro
            return price * 1.02  # 2% acima do preço atual
        
    def _find_previous_ll(self, price):
        """
        Encontra o Lower Low (LL) anterior para usar como stop loss.
        
        Args:
            price: Preço atual do ativo
            
        Returns:
            float: O menor preço (low) no período de lookback, ajustado se necessário
            
        Raises:
            ValueError: Se o preço for inválido ou não for possível determinar o LL
        """
        try:
            # Valida o preço
            if np.isnan(price) or np.isinf(price) or price <= 0:
                raise ValueError(f"Preço inválido: {price}")
                
            # Define o tamanho da janela de lookback
            lookback = min(20, self.current_step)
            if lookback <= 0:
                logging.warning("Lookback muito pequeno, usando valor padrão de 10")
                lookback = 10
                
            # Obtém a janela de dados
            start_idx = max(0, self.current_step - lookback)
            window = self.df.iloc[start_idx:self.current_step]
            
            # Verifica se a coluna 'low' existe
            if 'low' not in window.columns:
                logging.warning("Coluna 'low' não encontrada, usando 2% abaixo do preço")
                return price * 0.98
                
            # Encontra o menor low no período
            lowest_low = window['low'].min()
            
            # Valida o valor encontrado
            if np.isnan(lowest_low) or np.isinf(lowest_low) or lowest_low <= 0:
                logging.warning(f"Valor de lowest_low inválido: {lowest_low}, usando 2% abaixo do preço")
                return price * 0.98
                
            # Se o preço atual estiver muito próximo do menor low, 
            # usamos um valor padrão de stop loss (2% acima)
            if abs(lowest_low - price) / price < 0.01:  # menos de 1% de distância
                logging.debug(f"Preço muito próximo do menor low ({lowest_low:.2f}), usando 2% acima do preço")
                return price * 1.02
                
            logging.debug(f"Lower Low encontrado: {lowest_low:.2f} (preço atual: {price:.2f})")
            return lowest_low
            
        except Exception as e:
            logging.error(f"Erro ao encontrar Lower Low: {str(e)}")
            # Retorna um valor padrão seguro em caso de erro
            return price * 0.98  # 2% abaixo do preço atual
