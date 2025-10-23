# Como Coletar Dados para Treinamento - TraduLibras

Este guia explica como coletar dados para treinar novos modelos no sistema TraduLibras.

## 🎯 Objetivo da Coleta

Coletar gestos LIBRAS para treinar modelos de Machine Learning que reconheçam sinais específicos.

## 📝 Metodologia de Coleta

### 1. Preparação do Ambiente
```bash
# Instalar dependências
pip install -r requirements.txt

# Garantir que a câmera está funcionando
python -c "import cv2; cap = cv2.VideoCapture(0); print('Câmera OK' if cap.isOpened() else 'ERRO: Câmera não detectada')"
```

### 2. Coleta Manual Recomendada

#### **Para novas letras:**
1. Defina o conjunto de letras desejado
2. Calcule: `número de letras × 400 amostras`
3. Planeje sessões de coleta organizadas

### 3. Processo de Coleta

#### **Preparação:**
- Posicione a câmera em local fixo
- Garanta iluminação adequada
- Evite reflexos na mão
- Mantenha fundo neutro (preferencialmente branco/cinza)

#### **Durante a coleta:**
- Use apenas uma mão
- Mantenha o gesto estável por 1-2 segundos
- Varie ligeiramente a posição (pequenas variações naturais)
- Grave cada letra em diferentes ângulos
- Faça pausa entre gestos diferentes

#### **Qualidade dos dados:**
- ✅ Gestos claros e bem definidos
- ✅ Persistência por tempo adequado
- ✅ Variações naturais de posição
- ❌ Gestos muito rápidos ou confusos
- ❌ Múltiplas mãos
- ❌ Fundos ruidosos ou reflexos

## 🔄 Processamento dos Dados

### 1. Conversão para CSV
Converta as imagens coletadas para o formato CSV usado pelo sistema:

```bash
# Exemplo de script de conversão (personalize conforme necessário)
python -c "
import cv2
import mediapipe as mp
import os
import pandas as pd

# Seu código de processamento aqui
"

```

### 2. Formato CSV Esperado
```csv
gesture_type,x1,y1,x2,y2,...,x42,y42,x43,...,x51
A,-0.1,0.2,-0.05,0.15,...,0.3,-0.1,0.15,...,0.45
B,-0.08,0.18,-0.04,0.12,...,0.25,-0.08,0.12,...,0.42
...
```

### 3. Validação dos Dados
```bash
# Verificar estrutura dos dados
python -c "
import pandas as pd
df = pd.read_csv('gestos_libras.csv')
print(f'Total de amostras: {len(df)}')
print(f'Classes únicas: {df[\"gesture_type\"].unique()}')
print(f'Features por amostra: {df.shape[1]-1}')
"
```

## 🤖 Treinamento do Modelo

### 1. Processo de Treinamento
```bash
# Script de treinamento (adaptar conforme necessário)
python treinador_modelo_libras.py
```

### 2. Validação do Modelo
- **Acurácia esperada:** >80% no conjunto de teste
- **Cross-validation:** Use validação cruzada k-fold
- **Métricas:** Precisão, Recall, F1-Score por classe

## 💡 Dicas Importantes

### **Coleta Eficiente**
- Use cronômetro para manter tempo consistente por gesto
- Use grid de coleta (2-3 sessões de 5-10 minutos)
- Mantenha registro das condições de iluminação

### **Diversidade de Amostras**
- Grave com pessoas diferentes (principalmente objetivo final)
- Variações de tamanho de mão
- Pequenas variações de posicionamento natural

### **Controle de Quality**
- Revise periodicamente as amostras coletadas
- Remova amostras ruins antes do treinamento
- Esteja pronto para recolher dados problemáticos

### **Documentação**
- Mantenha registro das condições de coleta
- Documente classes e seu significado específico
- Registre limitações conhecidas do conjunto

## ❗ Considerações Importantes

1. **Licença de uso:** Certifique-se de ter permissão para coletar os dados
2. **LGPD:** Respeite leis de proteção de dados pessoais
3. **Consentimento:** Obtenha acordo explícito dos participantes
4. **Propósito:** Use os dados apenas para fins educacionais/inclusivos

## 🎓 Referências

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [LIBRAS Guidelines](https://www.gov.br/mdh/pt-br/assuntos/noticias/2023/diretrizes-da-politica-nacional-de-tradutores-interpretes-de-libras)
