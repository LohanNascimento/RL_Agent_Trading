import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import json
import os
import sys
import os

# Adiciona o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.binance_futures_testnet import BinanceFuturesTestnet

# Configuração do tema
THEME = {
    "primaryColor": "#7BD194",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA"
}

class TradingDashboard:
    def __init__(self):
        self.bot = BinanceFuturesTestnet()
        st.set_page_config(
            page_title="Crypto Trading Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_metrics(self):
        """Carrega métricas dos arquivos de log"""
        try:
            today_file = f"logs/status_{datetime.now().strftime('%Y%m%d')}.json"
            if os.path.exists(today_file):
                with open(today_file) as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar métricas: {str(e)}")
        return {}

    def display_account_summary(self):
        """Exibe resumo da conta"""
        summary = self.bot.get_account_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Saldo USDT", f"{summary['balance']:,.2f}$")
        with col2:
            st.metric("Exposição Total", f"{summary['exposure']:,.2f}$")
        with col3:
            st.metric("% Exposição", f"{summary['exposure_pct']:.2f}%")
        with col4:
            st.metric("Posições Ativas", summary['active_positions_count'])

    def display_positions(self):
        """Exibe posições ativas"""
        positions = self.bot.get_all_positions()
        if not positions:
            st.info("Nenhuma posição ativa no momento")
            return
        
        data = []
        for symbol, pos in positions.items():
            # Garante que pos é um dict (novo formato)
            if isinstance(pos, dict):
                size = pos['size']
                entry_price = pos['entry_price']
            else:
                # fallback para formato antigo (float)
                size = pos
                entry_price = 0.0

            current_price = self.bot.get_market_price(symbol)
            if current_price is None or entry_price == 0:
                pnl = 0.0
            elif size > 0:
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * abs(size)
            direction = 'LONG' if size > 0 else 'SHORT'
            data.append((symbol, abs(size), pnl, direction))

        df = pd.DataFrame(data, columns=['Símbolo', 'Tamanho', 'PnL', 'Direção'])

        st.subheader("🚀 Posições Abertas")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Símbolo": "Ativo",
                "Tamanho": st.column_config.NumberColumn(
                    "Contratos",
                    format="%.3f"
                ),
                "PnL": st.column_config.NumberColumn(
                    "PnL (USDT)",
                    format="%.2f",
                    help="Profit and Loss da posição em USDT"
                ),
                "Direção": "Direção"
            }
        )

    def display_order_history(self):
        """Exibe histórico de ordens"""
        metrics = self.load_metrics()
        if not metrics.get('trades_hoje'):
            st.info("Nenhuma ordem executada hoje")
            return
            
        st.subheader("📈 Histórico de Ordens")
        df = pd.DataFrame.from_dict(metrics['trades_hoje'], orient='index').reset_index()
        df.columns = ['Símbolo', 'Trades Hoje']
        st.bar_chart(df, x='Símbolo', y='Trades Hoje')

    def display_performance_chart(self):
        """Exibe gráfico de desempenho"""
        try:
            perf_files = sorted(os.listdir("../scripts/logs"), reverse=True)[:7]
            data = []
            
            for f in perf_files:
                if f.startswith('status'):
                    with open(f"../scripts/logs/{f}") as json_file:
                        data.append(json.load(json_file))
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                fig = px.line(df, x='timestamp', y='saldo', 
                            title='Evolução do Saldo')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível carregar dados de performance: {str(e)}")

    def run(self):
        """Executa o dashboard"""
        st.title("📊 Painel de Trading - Agente RL")
        
        # Atualização automática a cada 60 segundos
        st_autorefresh(interval=60 * 1000, key="data_refresh")
        
        with st.spinner("Atualizando dados..."):
            self.display_account_summary()
            self.display_performance_chart()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                self.display_positions()
            with col2:
                self.display_order_history()

def st_autorefresh(interval, key):
    """Hack para atualização automática"""
    import threading
    def reload():
        while True:
            time.sleep(interval/1000)
            st.experimental_rerun()
    
    if key not in st.session_state:
        st.session_state[key] = True
        thread = threading.Thread(target=reload)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()