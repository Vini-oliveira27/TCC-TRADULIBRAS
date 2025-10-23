@echo off
echo ========================================
echo INSTALACAO DO TRADULIBRAS EM NOVO PC
echo ========================================
echo.

echo [1/4] Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python nao encontrado!
    echo Instale Python 3.8+ de: https://python.org
    pause
    exit /b 1
)

echo [2/4] Criando ambiente virtual...
if exist venv (
    echo Ambiente virtual ja existe, removendo...
    rmdir /s /q venv
)
python -m venv venv

echo [3/4] Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo [4/4] Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo INSTALACAO CONCLUIDA COM SUCESSO!
echo ========================================
echo.
echo Para executar:
echo 1. Call venv\Scripts\activate.bat
echo 2. python app_funcional.py
echo.
echo Acesso: http://localhost:5000
echo Login: admin/admin123
echo ========================================
pause
