@echo off
REM Script para executar o sistema de trading em modo produção no Windows
REM Executa o modelo treinado na Binance Testnet com monitoramento contínuo

REM Configura variáveis de ambiente (preencha com suas credenciais)
set BINANCE_TESTNET_KEY=af48d3f1963b76c51f52066079b70af894b059d230ef2a850001f4f7e9431327
set BINANCE_TESTNET_SECRET=5169ec900dc7bb90ef5c28ab5db6ccd1eb1584080efca2ec4b41cd305b690a8b

REM Cria diretório para logs se não existir
if not exist ..\logs mkdir ..\logs

REM Data e hora atual para o arquivo de log
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"
set "LOG_FILE=..\logs\trading_live_%TIMESTAMP%.log"

echo Iniciando sistema de trading em %date% %time%
echo Logs serão salvos em: %LOG_FILE%

REM Executa o sistema de trading usando o novo arquivo de configuração de trade
python ..\scripts\run_testnet.py ^
  --config ..\config\trade_config.yaml

REM Para um trading real, remova a linha acima e descomente a linha abaixo (sem --monitor_only)
REM python ..\scripts\run_testnet.py ^
REM   --config ..\config\trade_config.yaml ^
REM   --monitor_only