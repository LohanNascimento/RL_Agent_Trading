import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import os
import sys
from PIL import Image
import plotly.io as pio

# Adiciona o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.binance_futures_testnet import BinanceFuturesTestnet

# Configuração do tema personalizado
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para melhorar o visual
def load_css():
    st.markdown("""
    <style>
        /* Cores e estilo geral */
        :root {
            --primary: #7BD194;
            --secondary: #6C63FF;
            --background: #0E1117;
            --card-bg: #1E2130;
            --text: #FFFFFF;
            --accent: #FF6B6B;
            --positive: #00C897;
            --negative: #FF5252;
        }
        
        /* Estilo do cabeçalho principal */
        .main-header {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
            text-align: center;
        }
        
        /* Cards para métricas */
        .metric-card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }
        
        .metric-title {
            font-size: 1rem;
            color: #AAAAAA;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .positive {
            color: var(--positive) !important;
        }
        
        .negative {
            color: var(--negative) !important;
        }
        
        /* Estilo para seções */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
        }
        
        /* Estilo para tabelas */
        .dataframe {
            border: none !important;
        }
        
        .dataframe th {
            background-color: var(--card-bg) !important;
            color: var(--text) !important;
            font-weight: 600 !important;
        }
        
        .dataframe td {
            background-color: var(--background) !important;
            color: var(--text) !important;
        }
        
        /* Remover bordas dos containers */
        div.block-container {
            padding-top: 2rem;
        }
        
        /* Estilo para o spinner de carregamento */
        div.stSpinner > div {
            border-top-color: var(--primary) !important;
        }
        
        /* Ajuste para gráficos Plotly */
        .js-plotly-plot .plotly {
            padding: 0 !important;
        }
        
        /* Estilo dos botões */
        .stButton > button {
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Estilos sidebar */
        .css-1d391kg {
            background-color: var(--card-bg);
        }
    </style>
    """, unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.bot = BinanceFuturesTestnet()
        load_css()
        self.previous_balance = None  # Armazena o saldo anterior para cálculo do delta
        self.balance_delta = 0.0  # Armazena a variação percentual do saldo
        self.load_data()
        self.init_sidebar()
        
    def load_data(self):
        """Carrega dados iniciais e calcula a variação do saldo"""
        current_summary = self.bot.get_account_summary()
        
        # Calcula a variação percentual do saldo
        if self.previous_balance is not None and self.previous_balance > 0:
            balance_change = ((current_summary['balance'] - self.previous_balance) / self.previous_balance) * 100
            self.balance_delta = round(balance_change, 2)
        
        # Atualiza o saldo anterior para a próxima iteração
        self.previous_balance = current_summary['balance']
        
        self.account_summary = current_summary
        self.positions = self.bot.get_all_positions()
        self.metrics = self.load_metrics()
        
    def init_sidebar(self):
        """Configura barra lateral"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center">
                <h1 style="color: #7BD194">Trading Bot</h1>
                <p>Agente de Trading com RL</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Atualização de dados
            st.subheader("📊 Controles")
            if st.button("🔄 Atualizar Dados", key="refresh_button"):
                with st.spinner("Atualizando..."):
                    self.load_data()
                    st.success("Dados atualizados!")
            
            # Exibe informações sobre o bot
            st.markdown("---")
            st.subheader("ℹ️ Informações")
            st.markdown("""
            <div style="font-size: 0.9rem;">
                <p>Última atualização: <span style="color: #7BD194; font-weight: 600;">
                    {}
                </span></p>
            </div>
            """.format(datetime.fromtimestamp(self.account_summary['timestamp']/1000).strftime('%H:%M:%S')), 
            unsafe_allow_html=True)
            
            # Configuração do painel
            st.markdown("---")
            st.subheader("⚙️ Configurações")
            self.auto_refresh = st.checkbox("Auto-atualização", value=True)
            if self.auto_refresh:
                self.refresh_interval = st.slider("Intervalo (segundos)", 
                                                 min_value=10, max_value=300, value=60, step=10)
                self.setup_auto_refresh(self.refresh_interval)
            
            st.markdown("---")
            st.markdown("<small>Desenvolvido com ❤️</small>", unsafe_allow_html=True)
        
    def load_metrics(self):
        """Carrega métricas dos arquivos de log"""
        try:
            today_file = f"/scripts/logs/status_{datetime.now().strftime('%Y%m%d')}.json"
            if os.path.exists(today_file):
                with open(today_file) as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar métricas: {str(e)}")
        return {}
        
    def create_metric_card(self, title, value, prefix="", suffix="", delta=None, delta_color="normal"):
        """Cria um card com métricas personalizadas"""
        delta_html = ""
        if delta is not None:
            delta_class = "positive" if delta > 0 else "negative" if delta < 0 else ""
            delta_icon = "↑" if delta > 0 else "↓" if delta < 0 else ""
            delta_html = f"""
            <div class="metric-delta {delta_class}">
                {delta_icon} {abs(delta):.2f}% desde ontem
            </div>
            """
            
        html = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
        """
        return html

    def display_account_summary(self):
        """Exibe resumo da conta com cards e gráficos"""
        # Linha de métricas principais
        summary = self.account_summary
        st.markdown('<h1 class="main-header">Painel de Trading</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(self.create_metric_card(
                "Saldo Total", 
                f"{summary['balance']:,.2f}", 
                suffix=" USDT",
                delta=self.balance_delta if hasattr(self, 'balance_delta') else 0.0
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(self.create_metric_card(
                "Exposição Total", 
                f"{summary['exposure']:,.2f}", 
                suffix=" USDT"
            ), unsafe_allow_html=True)
        
        with col3:
            exp_color = "normal"
            if summary['exposure_pct'] > 50:
                exp_color = "off"
            st.markdown(self.create_metric_card(
                "Exposição", 
                f"{summary['exposure_pct']:.2f}", 
                suffix="%",
                delta_color=exp_color
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(self.create_metric_card(
                "Posições Ativas", 
                f"{summary['active_positions_count']}"
            ), unsafe_allow_html=True)

    def display_positions(self):
        """Exibe posições ativas com gráfico de barras para PnL"""
        positions = self.positions
        
        st.markdown('<div class="section-header">🚀 Posições Abertas</div>', unsafe_allow_html=True)
        
        if not positions:
            st.info("Nenhuma posição ativa no momento")
            return
            
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Prepara dados para tabela
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
                    pnl_pct = 0.0
                elif size > 0:
                    pnl = (current_price - entry_price) * size
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                else:
                    pnl = (entry_price - current_price) * abs(size)
                    pnl_pct = ((entry_price / current_price) - 1) * 100
                
                direction = 'LONG' if size > 0 else 'SHORT'
                direction_icon = '📈' if size > 0 else '📉'
                price_change = f"{current_price - entry_price:.4f}"
                
                data.append({
                    "Símbolo": symbol,
                    "Dir": f"{direction_icon} {direction}",
                    "Tamanho": abs(size),
                    "Preço Entrada": entry_price,
                    "Preço Atual": current_price,
                    "Variação": price_change,
                    "PnL": pnl,
                    "PnL %": pnl_pct
                })

            df = pd.DataFrame(data)
            
            # Formatação condicional para PnL
            def color_pnl(val):
                try:
                    numeric_val = float(val)  # Convert to numeric
                    color = 'green' if numeric_val > 0 else 'red' if numeric_val < 0 else 'white'
                except ValueError:
                    color = 'white'  # Default color for non-numeric values
                return f"background-color: {color}"

            
            styled_df = df.style.applymap(color_pnl, subset=['PnL', 'PnL %', 'Variação'])
            
            # Exibe dataframe estilizado
            st.dataframe(
                styled_df,
                use_container_width=True,
                column_config={
                    "Símbolo": st.column_config.TextColumn("Ativo"),
                    "Dir": st.column_config.TextColumn("Direção"),
                    "Tamanho": st.column_config.NumberColumn("Contratos", format="%.3f"),
                    "Preço Entrada": st.column_config.NumberColumn("Entrada", format="%.4f"),
                    "Preço Atual": st.column_config.NumberColumn("Atual", format="%.4f"),
                    "Variação": st.column_config.NumberColumn("Var.", format="%.4f"),
                    "PnL": st.column_config.NumberColumn("PnL (USDT)", format="%.2f"),
                    "PnL %": st.column_config.NumberColumn("PnL %", format="%.2f")
                },
                hide_index=True
            )
        
        with col2:
            # Cria gráfico de barras para PnL
            if len(data) > 0:
                # Prepara dados para gráfico
                pnl_data = {
                    'Símbolo': [d['Símbolo'] for d in data],
                    'PnL': [d['PnL'] for d in data]
                }
                pnl_df = pd.DataFrame(pnl_data)
                
                # Cria gráfico de barras
                colors = ['#00C897' if pnl >= 0 else '#FF5252' for pnl in pnl_df['PnL']]
                
                fig = go.Figure(
                    go.Bar(
                        x=pnl_df['Símbolo'],
                        y=pnl_df['PnL'],
                        marker_color=colors,
                        text=pnl_df['PnL'].apply(lambda x: f"{x:.2f}"),
                        textposition='auto'
                    )
                )
                
                fig.update_layout(
                    title="Performance das Posições (PnL)",
                    title_x=0.5,
                    xaxis_title=None,
                    yaxis_title="PnL (USDT)",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(211, 211, 211, 0.2)'),
                    yaxis=dict(gridcolor='rgba(211, 211, 211, 0.2)'),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados para exibir o gráfico de PnL")

    def display_performance_chart(self):
        """Exibe gráfico de desempenho com melhorias visuais"""
        st.markdown('<div class="section-header">📈 Evolução do Saldo</div>', unsafe_allow_html=True)
        
        try:
            # Carregar dados de exemplo se os logs não estiverem disponíveis
            perf_files = sorted(os.listdir("../logs/"), reverse=True)[:7]
            data = []
            
            for f in perf_files:
                if f.startswith('status'):
                    with open(f"/scripts/logs/{f}") as json_file:
                        data.append(json.load(json_file))
            
            # Se não houver dados reais, use dados de exemplo para demonstração
            if not data:
                # Dados de exemplo
                example_data = [
                    {"timestamp": "2025-05-01T00:00:00", "saldo": 1000.0},
                    {"timestamp": "2025-05-02T00:00:00", "saldo": 1035.0},
                    {"timestamp": "2025-05-03T00:00:00", "saldo": 1072.0},
                    {"timestamp": "2025-05-04T00:00:00", "saldo": 1063.0},
                    {"timestamp": "2025-05-05T00:00:00", "saldo": 1098.0},
                    {"timestamp": "2025-05-06T00:00:00", "saldo": 1127.0},
                    {"timestamp": "2025-05-07T00:00:00", "saldo": 1151.0},
                    {"timestamp": "2025-05-08T00:00:00", "saldo": 1172.0},
                    {"timestamp": "2025-05-09T00:00:00", "saldo": 1190.0},
                    {"timestamp": "2025-05-10T00:00:00", "saldo": 1215.0},
                    {"timestamp": "2025-05-11T00:00:00", "saldo": 1250.0},
                ]
                data = example_data
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Cálculo de métricas relacionadas ao desempenho
            initial_balance = df.iloc[0]['saldo'] if not df.empty else 0
            current_balance = df.iloc[-1]['saldo'] if not df.empty else 0
            performance = ((current_balance / initial_balance) - 1) * 100 if initial_balance > 0 else 0
            
            # Exibe o gráfico e métricas
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Gráfico de área
                fig = go.Figure()
                
                # Área sombreada
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['saldo'],
                        fill='tozeroy',
                        fillcolor='rgba(0, 200, 151, 0.2)',
                        line=dict(color='#00C897', width=2),
                        mode='lines',
                        name='Saldo'
                    )
                )
                
                # Adiciona linha de tendência
                fig.add_trace(
                    go.Scatter(
                        x=[df['timestamp'].min(), df['timestamp'].max()],
                        y=[initial_balance, current_balance],
                        mode='lines',
                        line=dict(dash='dash', color='#6C63FF', width=1),
                        name='Tendência'
                    )
                )
                
                # Estilo do gráfico
                fig.update_layout(
                    template="plotly_dark",
                    title=None,
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title=None,
                        showgrid=True,
                        gridcolor='rgba(211, 211, 211, 0.2)'
                    ),
                    yaxis=dict(
                        title=None,
                        showgrid=True,
                        gridcolor='rgba(211, 211, 211, 0.2)',
                        tickformat=',.0f'
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Exibe o gráfico
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Métricas de performance
                st.markdown(self.create_metric_card(
                    "Performance", 
                    f"{performance:.2f}", 
                    suffix="%",
                    delta=performance
                ), unsafe_allow_html=True)
                
                st.markdown(self.create_metric_card(
                    "Saldo Inicial", 
                    f"{initial_balance:.2f}", 
                    suffix=" USDT"
                ), unsafe_allow_html=True)
                
                st.markdown(self.create_metric_card(
                    "Saldo Atual", 
                    f"{current_balance:.2f}", 
                    suffix=" USDT"
                ), unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Não foi possível carregar dados de performance: {str(e)}")

    def display_market_overview(self):
        """Exibe visão geral do mercado com métricas por símbolo"""
        st.markdown('<div class="section-header">📊 Visão do Mercado</div>', unsafe_allow_html=True)
        
        # Verifica se há métricas disponíveis
        if not self.metrics.get('trades_hoje'):
            # Dados de exemplo para demonstração
            symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT']
            market_data = {
                'Símbolo': symbols,
                'Preço': [65820.5, 3075.4, 0.528, 0.48, 6.37],
                'Variação 24h': [2.4, -1.3, 0.7, -0.5, 1.2],
                'Volume': [12.5, 8.7, 3.2, 2.1, 1.8],
                'Trades': [24, 18, 12, 7, 9]
            }
        else:
            # Dados reais
            trades_hoje = self.metrics.get('trades_hoje', {})
            symbols = list(trades_hoje.keys())
            market_data = {
                'Símbolo': symbols,
                'Preço': [self.bot.get_market_price(sym) for sym in symbols],
                'Variação 24h': [1.2, -0.8, 0.5, -0.3, 0.9][:len(symbols)],  # Valores de exemplo
                'Volume': [5.2, 3.7, 2.1, 1.5, 1.2][:len(symbols)],  # Valores de exemplo
                'Trades': list(trades_hoje.values())
            }
        
        df = pd.DataFrame(market_data)
        
        # Exibe duas colunas: gráfico de pie e tabela
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gráfico de pizza para volume de trades por símbolo
            fig = go.Figure(
                go.Pie(
                    labels=df['Símbolo'],
                    values=df['Trades'],
                    hole=0.5,
                    marker=dict(
                        colors=['#00C897', '#6C63FF', '#FF6B6B', '#FFAB4C', '#845EC2']
                    ),
                    textinfo='percent',
                    hoverinfo='label+value+percent'
                )
            )
            
            fig.update_layout(
                title="Distribuição de Trades",
                title_x=0.5,
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabela de dados de mercado
            # Formatação condicional para variação
            def color_var(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'white'
                return f'color: {color}'
            
            styled_df = df.style.applymap(color_var, subset=['Variação 24h'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                column_config={
                    "Símbolo": st.column_config.TextColumn("Ativo"),
                    "Preço": st.column_config.NumberColumn("Preço Atual", format="%.4f"),
                    "Variação 24h": st.column_config.NumberColumn("Var. 24h (%)", format="%.2f"),
                    "Volume": st.column_config.NumberColumn("Volume (M)", format="%.1f"),
                    "Trades": st.column_config.NumberColumn("Trades", format="%d")
                },
                hide_index=True
            )

    def setup_auto_refresh(self, seconds=60):
        """Configura atualização automática do dashboard"""
        st_autorefresh(seconds * 1000, f"autorefresh_{seconds}")
        
    def run(self):
        """Executa o dashboard"""
        # Estrutura das seções
        self.display_account_summary()
        self.display_performance_chart()
        
        # Posições e mercado lado a lado
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self.display_positions()
        
        with col2:
            self.display_market_overview()

# Função de auto-refresh (mantido do código original)
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