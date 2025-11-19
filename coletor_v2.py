import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Frases para treinar
FRASES = [
    "abcdefghijklmnopqrstuvwxyz",
    "TraduLibras"
]

# Extrair letras únicas das frases (convertendo para maiúsculas) e adicionar ESPAÇO e PONTO
letras_base = set(''.join(FRASES).upper().replace(' ', ''))
letras = sorted(letras_base) + ['ESPACO', 'PONTO']  # Adicionando os gestos especiais

print("\nGestos que serão treinados:")
print(letras)
print(f"Total de gestos únicos: {len(letras)}")

# Configurações
AMOSTRAS_POR_GESTO = 200  # Número de amostras para cada gesto

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def extrair_caracteristicas(landmarks):
    """Extrai as características dos pontos de referência da mão"""
    p0 = landmarks.landmark[0]  # Ponto de referência do pulso
    dados = []
    for lm in landmarks.landmark:
        dados.extend([
            lm.x - p0.x,
            lm.y - p0.y,
            lm.z - p0.z
        ])
    return dados

def coletar_dados():
    """Coleta dados para todos os gestos necessários"""
    dados_treinamento = []
    labels = []
    
    # Criar diretório para o modelo se não existir
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
    
    camera = cv2.VideoCapture(0)
    
    for gesto in letras:
        amostras_coletadas = 0
        print(f"\n=== Coletando dados para o gesto '{gesto}' ===")
        print(f"Objetivo: {AMOSTRAS_POR_GESTO} amostras")
        
        # Instruções específicas para cada gesto
        if gesto == 'ESPACO':
            print("💡 Gesto ESPAÇO: Faça um gesto representando espaço (ex: mão aberta movendo para o lado)")
        elif gesto == 'PONTO':
            print("💡 Gesto PONTO: Faça um gesto representando ponto final (ex: punho fechado)")
        else:
            print(f"💡 Letra {gesto}: Faça o gesto da letra {gesto} em LIBRAS")
        
        print("Pressione 'ESPAÇO' para capturar uma amostra")
        print("Pressione 'ESC' para pular este gesto")
        
        while amostras_coletadas < AMOSTRAS_POR_GESTO:
            success, frame = camera.read()
            if not success:
                continue
            
            # Espelha o frame para uma experiência mais natural
            frame = cv2.flip(frame, 1)
            
            # Processa o frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Desenha as informações na tela
            cv2.putText(frame, f"Gesto: {gesto}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Amostras: {amostras_coletadas}/{AMOSTRAS_POR_GESTO}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Dica visual para gestos especiais
            if gesto == 'ESPACO':
                cv2.putText(frame, "Dica: Mova a mao para o lado", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif gesto == 'PONTO':
                cv2.putText(frame, "Dica: Punho fechado", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Se detectou mão, desenha os pontos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow('Coleta de Dados - TraduLibras', frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print(f"⏭️  Pulando gesto {gesto}")
                break
            elif key == 32 and results.multi_hand_landmarks:  # ESPAÇO
                # Extrai características e salva
                for hand_landmarks in results.multi_hand_landmarks:
                    caracteristicas = extrair_caracteristicas(hand_landmarks)
                    if len(caracteristicas) == 63:  # 21 pontos * 3 coordenadas
                        dados_treinamento.append(caracteristicas)
                        labels.append(gesto)
                        amostras_coletadas += 1
                        print(f"✓ {gesto} - Amostra {amostras_coletadas} coletada!")
            
            if amostras_coletadas >= AMOSTRAS_POR_GESTO:
                print(f"✅ Coleta para {gesto} finalizada!")
                break
    
    camera.release()
    cv2.destroyAllWindows()
    
    return np.array(dados_treinamento), np.array(labels)

def treinar_modelo(X, y):
    """Treina o modelo com os dados coletados"""
    print("\n🤖 Treinando o modelo...")
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Cria e treina o modelo
    modelo = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=15,
        min_samples_split=5
    )
    modelo.fit(X_train, y_train)
    
    # Avalia o modelo
    acuracia_treino = modelo.score(X_train, y_train)
    acuracia_teste = modelo.score(X_test, y_test)
    
    print(f"\n📊 Resultados do Treinamento:")
    print(f"   Acurácia no treino: {acuracia_treino:.2%}")
    print(f"   Acurácia no teste:  {acuracia_teste:.2%}")
    
    # Mostrar distribuição das classes
    from collections import Counter
    distribuição = Counter(y)
    print(f"\n📈 Distribuição dos dados:")
    for gesto, count in distribuição.items():
        print(f"   {gesto}: {count} amostras")
    
    return modelo

def main():
    print("🎯 Treinamento de Reconhecimento de LIBRAS - TraduLibras")
    print("=" * 50)
    print("\nFrases de referência:")
    for frase in FRASES:
        print(f"- {frase}")
    print(f"\n➕ Gestos adicionais: ESPAÇO, PONTO")
    
    print("\n💡 DICAS PARA COLETA:")
    print("   - ESPAÇO: Gesto para espaço entre palavras")
    print("   - PONTO: Gesto para ponto final")
    print("   - Mantenha cada gesto por 2-3 segundos")
    print("   - Use boa iluminação e fundo uniforme")
    print("=" * 50)
    
    input("\nPressione ENTER para começar a coleta de dados...")
    
    # Coleta os dados
    X, y = coletar_dados()
    
    if len(X) > 0:
        print(f"\n📦 Dados coletados: {len(X)} amostras")
        
        # Treina o modelo
        modelo = treinar_modelo(X, y)
        
        # Salva o modelo
        with open('modelos/modelo_libras.pkl', 'wb') as f:
            pickle.dump(modelo, f)
        print("\n💾 Modelo salvo com sucesso em 'modelos/modelo_libras.pkl'")
        
        # Salva os gestos treinados
        with open('modelos/gestos_treinados.txt', 'w') as f:
            f.write(','.join(letras))
        print("💾 Gestos treinados salvos em 'modelos/gestos_treinados.txt'")
        
        # Salva informações adicionais
        info_modelo = {
            'gestos_treinados': letras,
            'total_amostras': len(X),
            'data_treinamento': np.datetime64('now').astype(str),
            'tipo_modelo': 'RandomForest',
            'features': 63
        }
        
        with open('modelos/info_modelo.pkl', 'wb') as f:
            pickle.dump(info_modelo, f)
        print("💾 Informações do modelo salvas em 'modelos/info_modelo.pkl'")
        
        print("\n🎉 Treinamento concluído com sucesso!")
        print("🚀 Agora execute: python app.py")
        
    else:
        print("\n❌ Nenhum dado coletado. O modelo não foi treinado.")

if __name__ == "__main__":
    main()