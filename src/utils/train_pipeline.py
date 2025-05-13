"""
Módulo de pipeline de treinamento para modelos de aprendizado por reforço.

Este módulo fornece funções utilitárias para preparar dados, normalizar features
e treinar modelos de RL de forma padronizada.
"""
import logging
from typing import Tuple, List, Optional, Union, Dict, Any, TypeVar

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

# Tipo genérico para o modelo de RL
RLModel = TypeVar('RLModel', bound=BaseAlgorithm)

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    split_ratios: Tuple[float, float],
    shuffle: bool = False,
    random_state: Optional[int] = None,
    time_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepara os dados para treino, validação e teste, mantendo a ordem temporal.
    
    Args:
        df: DataFrame com os dados
        feature_cols: Lista de colunas de features
        split_ratios: Tupla com as proporções (train_ratio, val_ratio)
        shuffle: Se True, embaralha os dados antes da divisão (não recomendado para dados temporais)
        random_state: Semente para reprodutibilidade
        
    Returns:
        Tuple com DataFrames de treino, validação e teste
        
    Raises:
        ValueError: Se as proporções forem inválidas ou se não houver dados suficientes
    """
    train_ratio, val_ratio = split_ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Validação das proporções
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or not (0 <= test_ratio < 1):
        raise ValueError("As proporções devem estar entre 0 e 1")
    
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-10:
        raise ValueError(f"A soma das proporções deve ser igual a 1, mas é {train_ratio + val_ratio + test_ratio}")
    
    # Ordena por tempo se uma coluna de tempo for fornecida
    if time_col and time_col in df.columns:
        logger.info(f"Ordenando dados pela coluna temporal: {time_col}")
        df = df.sort_values(by=time_col)
    elif not shuffle:
        logger.warning("Nenhuma coluna temporal fornecida e shuffle=False. "
                     "Certifique-se de que os dados já estão na ordem correta.")
    
    if len(df) < 3:  # Mínimo de 3 amostras para ter pelo menos 1 em cada conjunto
        raise ValueError("Número insuficiente de amostras para divisão")
    
    logger.info(f"Preparando dados: total={len(df)} amostras, "
                f"treino={train_ratio*100:.1f}%, "
                f"validação={val_ratio*100:.1f}%, "
                f"teste={test_ratio*100:.1f}%")
    
    # Verifica colunas ausentes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas de features não encontradas no DataFrame: {missing_cols}")
    
    # Divisão dos dados
    if shuffle:
        logger.info("Embaralhando dados antes da divisão")
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calcula índices de corte
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    
    # Garante que temos pelo menos uma amostra em cada conjunto
    n_train = max(1, min(n_train, n - 2))
    n_val = max(n_train + 1, min(n_val, n - 1))
    
    # Divide os dados
    train_data = df.iloc[:n_train].copy()
    val_data = df.iloc[n_train:n_val].copy()
    test_data = df.iloc[n_val:].copy()
    
    # Verifica se os conjuntos não estão vazios
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("Um ou mais conjuntos de dados ficaram vazios após a divisão. "
                        f"Tamanhos: treino={len(train_data)}, val={len(val_data)}, teste={len(test_data)}")
    
    logger.info(f"Divisão concluída: treino={len(train_data)}, "
                f"validação={len(val_data)}, teste={len(test_data)}")
    
    return train_data, val_data, test_data

def normalize_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normaliza as features usando StandardScaler.
    
    Args:
        train_df: DataFrame de treino
        val_df: DataFrame de validação
        test_df: DataFrame de teste
        feature_cols: Lista de colunas a serem normalizadas
        scaler: Instância opcional de StandardScaler. Se None, um novo será criado.
        
    Returns:
        Tupla com (train_df, val_df, test_df, scaler)
        
    Raises:
        ValueError: Se não houver colunas numéricas para normalizar
    """
    # Filtra apenas colunas numéricas presentes
    numeric_cols = [
        col for col in feature_cols 
        if col in train_df.columns and 
        pd.api.types.is_numeric_dtype(train_df[col])
    ]
    
    if not numeric_cols:
        logger.warning("Nenhuma coluna numérica encontrada para normalização")
        return train_df, val_df, test_df, None
    
    logger.info(f"Normalizando {len(numeric_cols)} colunas numéricas")
    
    # Cria ou usa o scaler fornecido
    if scaler is None:
        scaler = StandardScaler()
        train_values = train_df[numeric_cols].values
        scaler.fit(train_values)
    
    # Aplica a transformação
    def safe_transform(df, cols, scaler):
        if df is None or len(df) == 0:
            return df
        df = df.copy()
        df[cols] = scaler.transform(df[cols].values)
        return df
    
    train_df = train_df.copy()
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols].values)
    val_df = safe_transform(val_df, numeric_cols, scaler)
    test_df = safe_transform(test_df, numeric_cols, scaler)
    
    return train_df, val_df, test_df, scaler

def train_model(
    env: Union[gym.Env, VecEnv],
    model: RLModel,
    total_timesteps: int,
    callbacks: Optional[Union[Any, List[Any]]] = None,
    progress_bar: bool = True,
    tb_log_name: str = "rl_training",
    reset_num_timesteps: bool = True,
    **kwargs
) -> RLModel:
    """
    Treina o modelo em um ambiente especificado.
    
    Args:
        env: Ambiente de treinamento (Gym ou VecEnv)
        model: Modelo de RL a ser treinado
        total_timesteps: Número total de passos de treinamento
        callbacks: Lista de callbacks para o treinamento
        progress_bar: Se True, exibe barra de progresso
        tb_log_name: Nome para o log do TensorBoard
        reset_num_timesteps: Se True, reinicia o contador de passos
        **kwargs: Argumentos adicionais para o método learn()
        
    Returns:
        Modelo treinado
        
    Raises:
        TypeError: Se o ambiente ou modelo não forem do tipo esperado
        ValueError: Se houver problemas com os parâmetros
    """
    import gymnasium as gym
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    
    # Validação de tipos
    if not isinstance(env, (gym.Env, VecEnv)):
        raise TypeError(f"O ambiente deve ser do tipo gym.Env ou VecEnv, não {type(env)}")
    
    if not isinstance(model, BaseAlgorithm):
        raise TypeError(f"O modelo deve ser uma instância de BaseAlgorithm, não {type(model)}")
    
    if not isinstance(total_timesteps, int) or total_timesteps <= 0:
        raise ValueError(f"total_timesteps deve ser um inteiro > 0, não {total_timesteps}")
    
    # Prepara callbacks
    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]
        
    # Verifica se os callbacks são válidos
    valid_callbacks = []
    for cb in callbacks:
        if hasattr(cb, '__call__') or hasattr(cb, 'on_step'):
            valid_callbacks.append(cb)
        else:
            logger.warning(f"Callback {cb} não é um callback válido e será ignorado")
    callbacks = valid_callbacks
    
    # Configura callbacks
    if len(callbacks) > 1:
        callback = CallbackList(callbacks)
    elif callbacks:
        callback = callbacks[0]
    else:
        callback = None
    
    # Log de informações
    logger.info(f"Iniciando treinamento para {total_timesteps} timesteps")
    logger.info(f"Usando callbacks: {[cb.__class__.__name__ for cb in (callbacks or [])]}")
    
    # Treina o modelo
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs
        )
        logger.info("Treinamento concluído com sucesso")
        return model
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise
