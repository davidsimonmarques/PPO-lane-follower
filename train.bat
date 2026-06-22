@echo off
REM ============================================================
REM  Script de Treinamento PPO - Lane Follower com CARLA
REM  
REM  Otimizado para: AMD RX6600 8GB + 16GB RAM + Xeon E5-2666v3
REM  
REM  1. Inicia 1 instância CARLA (offscreen)
REM  2. Inicia TensorBoard
REM  3. Executa treinamento PPO
REM  4. Encerra CARLA ao finalizar
REM ============================================================

REM --- Caminhos ---
set CARLA_EXE=C:\CARLA\CarlaUE4.exe
set PYTHON=python
set PROJECT_DIR=%~dp0
set SRC_DIR=%PROJECT_DIR%src

echo ============================================================
echo  PPO Lane Follower - Treinamento Otimizado
echo ============================================================
echo  Hardware: AMD RX6600 8GB + 16GB RAM
echo  Modo: DummyVecEnv (1 instancia, minimo de processamento)
echo.

REM ============================================================
REM  1. INICIAR CARLA
REM ============================================================
echo [1/4] Iniciando CARLA (offscreen)...
start "CARLA" /B "%CARLA_EXE%" -carla-rpc-port=2000 -RenderOffScreen -nosound

echo  Aguardando CARLA inicializar (20 segundos)...
ping -n 21 127.0.0.1 >nul 2>&1
echo  CARLA pronto.
echo.

REM ============================================================
REM  2. INICIAR TENSORBOARD
REM ============================================================
echo [2/4] Iniciando TensorBoard...
if not exist "%PROJECT_DIR%logs\tensorboard" mkdir "%PROJECT_DIR%logs\tensorboard"
start "TensorBoard" cmd /k "tensorboard --logdir=%PROJECT_DIR%\src\logs\tensorboard --port=6006"
echo  TensorBoard: http://localhost:6006
echo.

ping -n 4 127.0.0.1 >nul 2>&1

REM ============================================================
REM  3. EXECUTAR TREINAMENTO
REM ============================================================
echo [3/4] Iniciando treinamento PPO...
echo ============================================================
echo.

cd /d "%SRC_DIR%"
%PYTHON% main.py

echo.
echo ============================================================
echo  Treinamento finalizado.
echo ============================================================
echo.

REM ============================================================
REM  4. ENCERRAR CARLA
REM ============================================================
echo [4/4] Encerrando CARLA...
taskkill /F /IM CarlaUE4.exe >nul 2>&1
taskkill /F /IM CarlaUE4-Win64-Shipping.exe >nul 2>&1
echo  CARLA encerrado.
echo.

echo Pressione qualquer tecla para fechar...
pause >nul