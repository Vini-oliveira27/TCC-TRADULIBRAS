#!/bin/bash

echo "========================================"
echo "INSTALAÇÃO DO TRADULIBRAS EM NOVO PC"
echo "========================================"
echo

echo "[1/5] Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Instale Python 3.8+ de: https://python.org"
    exit 1
fi

echo "[2/5] Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    echo "ERRO: pip3 não encontrado!"
    echo "Instale pip3 primeiro"
    exit 1
fi

echo "[3/5] Criando ambiente virtual..."
if [ -d "venv" ]; then
    echo "Ambiente virtual já existe, removendo..."
    rm -rf venv
fi
python3 -m venv venv

echo "[4/5] Ativando ambiente virtual..."
source venv/bin/activate

echo "[5/5] Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "========================================"
echo "INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
echo "========================================"
echo
echo "Para executar:"
echo "1. source venv/bin/activate"
echo "2. python app_funcional.py"
echo
echo "Acesso: http://localhost:5000"
echo "Login: admin/admin123"
echo "========================================"
