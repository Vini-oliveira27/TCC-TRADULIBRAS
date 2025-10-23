# Guia de Replicação - TraduLibras

Este guia explica como replicar o projeto TraduLibras em outro computador/notebook.

## 🚀 Replicação Rápida (Windows)

### Opção 1: Script Automático
1. **Copie todos os arquivos** do projeto para o novo PC
2. **Execute o instalador:**
   ```cmd
   INSTALAR.bat
   ```
3. **Aguarde a instalação** automática
4. **Execute a aplicação:**
   ```cmd
   venv\Scripts\activate.bat
   python app_funcional.py
   ```

### Opção 2: Manual
```cmd
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
venv\Scripts\activate.bat

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
python app_funcional.py
```

## 🚀 Replicação Rápida (Linux/Mac)

### Opção 1: Script Automático
1. **Copie todos os arquivos** do projeto para o novo PC
2. **Torne executável e execute:**
   ```bash
   chmod +x INSTALAR.sh
   ./INSTALAR.sh
   ```
3. **Execute a aplicação:**
   ```bash
   source venv/bin/activate
   python app_funcional.py
   ```

### Opção 2: Manual
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
python app_funcional.py
```

## 📋 Requisitos do Sistema

### Hardware Mínimo
- **Processador:** Dual-core 2.0GHz+
- **RAM:** 4GB+
- **Espaço:** 2GB livres
- **Câmera:** Webcam USB ou integrada

### Software Necessário
- **Python:** 3.8 ou superior
- **OS:** Windows 10+, macOS 10.15+, ou Ubuntu 18.04+

## 🎯 Checklist de Replicação

### ✅ Arquivos Essenciais
- [ ] `app_funcional.py` - Aplicação principal
- [ ] `auth.py` - Sistema de autenticação
- [ ] `requirements.txt` - Dependências
- [ ] `users.json` - Dados dos usuários (é criado automaticamente)
- [ ] `modelos/` - Diretório com 3 arquivos .pkl
- [ ] `templates/` - Templates HTML (9 arquivos)
- [ ] `static/` - Arquivos estáticos (CSS + imagens)

### ✅ Funcionalidades a Testar
- [ ] **Instalação:** Script executa sem erros
- [ ] **Câmera:** Detecta e inicializa webcam
- [ ] **Interface:** Acessa http://localhost:5000
- [ ] **Login:** Consegue fazer login com admin/admin123 ou user/user123
- [ ] **Reconhecimento:** Detecta gestos na câmera
- [ ] **Áudio:** Reproduz texto sintético
- [ ] **Rede:** Funciona em http://IP:5000

## 🚨 Solução de Problemas

### Problema: Python não encontrado
```bash
# Windows
# Instalar Python: https://python.org
# Marcar "Add Python to PATH"

# Linux
sudo apt install python3 python3-pip

# macOS
brew install python3
```

### Problema: PIP não encontrado
```bash
# Windows
python -m ensurepip --upgrade

# Linux/macOS
python3 -m ensurepip<｜tool▁sep｜>upgrade
```

### Problema: Câmera não detectada
```bash
# Verificar permissões (Linux/macOS)
ls /dev/video*

# Verificar drivers (Windows)
# Abrir Gerenciador de Dispositivos -> Imagens
```

### Problema: Porta 5000 bloqueada
```bash
# Verificar processo usando porta
netstat -an | findstr :5000

# Usar porta diferente
python app_funcional.py --port 8000
```

### Problema: Firewall bloqueando rede
- **Windows:** Adicionar exceção para Python na porta 5000
- **Linux:** `sudo ufw allow 5000`
- **macOS:** Configurações do Sistema -> Rede -> Firewall

## 📦 Comando de Deploy Rápido

### Para Windows:
```batch
@echo off
echo Copiando arquivos...
xcopy /E /I /H /Y "origem" "destino"
cd "destino"
call INSTALAR.bat
```

### Para Linux:
```bash
#!/bin/bash
echo "Copiando arquivos..."
cp -r origem/* destino/
cd destino
chmod +x INSTALAR.sh
./INSTALAR.sh
```

## ✨ Status: PRONTO PARA REPLICACAO!

O projeto está **100% preparado** para replicação com:
- ✅ Scripts de instalação automática
- ✅ Documentação completa
- ✅ Checklist de verificação
- ✅ Soluções para problemas comuns
- ✅ Estrutura limpa e organizada

**Tempo médio de replicação:** 5-10 minutos
