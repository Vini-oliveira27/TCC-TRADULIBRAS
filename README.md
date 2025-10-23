# TraduLibras - Reconhecimento de Gestos LIBRAS

Sistema funcional de reconhecimento de gestos em LIBRAS com interface web estável.

## 🚀 Instalação e Execução Rápida

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar aplicação (PRONTO PARA USO)
```bash
python app_funcional.py
```

### 3. Acessar sistema
- **Local:** http://localhost:5000
- **Rede:** http://SEU_IP:5000

#### 🔐 Credenciais de acesso:
- **Admin:** `admin` / `admin123`
- **Usuário:** `user` / `user123`

## 📋 Funcionalidades

### ✅ Sistema Atual 
- **Reconhecimento em tempo real** das letras de A-Z
- **Interface web responsiva** com câmera e sistema de controle da mão robótica integrada
- **Síntese de voz** integrada (gTTS)
- **Sistema de autenticação** com níveis de acesso
- **Detecção estável** com cooldown inteligente

### 🔧 Especificações Técnicas
- **Modelo:** Ensemble (Random Forest + SVM + KNN)
- **Features:** 51 landmarks + distâncias otimizadas
- **Frameworks:** Flask + MediaPipe + OpenCV
- **Voz:** Google Text-to-Speech (gTTS)

## 📁 Estrutura do Projeto

## 🌐 Acesso na Rede Local

1. **Todos os dispositivos devem estar na mesma rede Wi-Fi**
2. **Use o IP mostrado no terminal:** `http://IP_DO_COMPUTADOR:5000`
3. **Exemplo:** `http://192.168.1.100:5000`

## 📊 Status do Sistema

O sistema está **100% funcional** e optimizado:
- ✅ Modelos treinados e carregados
- ✅ Câmera funcionando
- ✅ Reconhecimento estável
- ✅ Interface responsiva
- ✅ Áudio integrado
- ✅ Autenticação segura

## 📚 Documentação Adicional

- **Instruções de coleta:** Ver `COMO_COLETAR_DADOS.md`
- **Desenvolvimento:** Projeto limpo e organizado
- **Versão:** 2.0.0 - FUNCIONAL

## 🏆 Pronto para Produção!

Este sistema está **estavel, testeado e pronto para uso** em ambientes educacionais ou de inclusão social!