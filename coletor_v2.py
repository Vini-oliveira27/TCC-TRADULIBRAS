#!/usr/bin/env python3
"""
COLETOR LIBRAS - VERSÃO COMPATÍVEL COM APP.PY
Coletor otimizado para o sistema TraduLibras
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime

class ColetorLIBRAS:
    def __init__(self, pasta_dados='dados_libras'):
        self.pasta_dados = pasta_dados
        self.arquivo_csv = 'dataset_libras.csv'
        self.caminho_arquivo = os.path.join(pasta_dados, self.arquivo_csv)
        
        # MediaPipe (mesma configuração do app.py)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Criar pasta se não existir
        if not os.path.exists(pasta_dados):
            os.makedirs(pasta_dados)
        
        # Dados
        self.dados = []
        self.classe_atual = "A"
        self.contador = 0
        self.total_amostras = 0
        
        # Configurações
        self.cooldown = 0.5  # segundos entre amostras
        self.ultima_coleta = 0
        self.coletando = False
        
        # Classes suportadas (mesmas do app.py)
        self.classes_suportadas = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ESPACO', 'PONTO'
        ]
        
        print("🚀 COLETOR LIBRAS - TRADULIBRAS")
        print("=" * 50)
        print(f"📁 Pasta de dados: {self.pasta_dados}")
        print(f"📊 Classes: {self.classes_suportadas}")
        print("=" * 50)
    
    def extrair_features(self, hand_landmarks):
        """Extrair 51 features - IDÊNTICO AO APP.PY"""
        if not hand_landmarks:
            return None
            
        wrist = hand_landmarks.landmark[0]
        features = []
        
        # 1. Coordenadas relativas ao pulso (42 features)
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)  # 21 x
            features.append(lm.y - wrist.y)  # 21 y
        
        # Pontas dos dedos
        tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
        
        # 2. Distâncias das pontas ao pulso (5 features)
        features += [abs(tip.x - wrist.x) + abs(tip.y - wrist.y) for tip in tips]
        
        # 3. Distâncias entre dedos consecutivos (4 features)
        features += [abs(tips[i].x - tips[i+1].x) + abs(tips[i].y - tips[i+1].y) for i in range(4)]
        
        return features  # Total: 42 + 5 + 4 = 51 features
    
    def detectar_webcam_usb(self):
        """Detectar webcam USB automaticamente"""
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened() and cap.read()[0]:
                    print(f"📹 Webcam detectada: índice {i}")
                    cap.release()
                    return i
            except:
                pass
        print("⚠️  Usando webcam padrão (índice 0)")
        return 0
    
    def mostrar_estatisticas(self, frame):
        """Mostrar estatísticas na tela"""
        # Fundo para informações
        cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
        
        # Classe atual
        cv2.putText(frame, f"CLASSE: {self.classe_atual}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Contadores
        cv2.putText(frame, f"SESSÃO: {self.contador}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"TOTAL: {self.total_amostras}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Status
        status_color = (0, 255, 0) if self.coletando else (0, 0, 255)
        status_text = "✅ COLETANDO" if self.coletando else "⏸️ PAUSADO"
        cv2.putText(frame, status_text, (450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Instruções
        cv2.putText(frame, "A/Z: Mudar classe | ESPACO: Coletar/Pausar", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "S: Salvar | Q: Sair | R: Resetar sessao", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Cooldown: {self.cooldown}s", (10, 490), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def carregar_estatisticas(self):
        """Carregar estatísticas existentes"""
        if os.path.exists(self.caminho_arquivo):
            df = pd.read_csv(self.caminho_arquivo)
            self.total_amostras = len(df)
            print(f"📊 Dataset existente: {self.total_amostras} amostras")
            
            # Mostrar distribuição
            distribuição = df['classe'].value_counts()
            print("📈 Distribuição atual:")
            for classe, count in distribuição.items():
                print(f"   {classe}: {count} amostras")
        else:
            print("📝 Criando novo dataset")
    
    def salvar_dados(self):
        """Salvar dados coletados"""
        if not self.dados:
            print("❌ Nenhum dado para salvar")
            return False
        
        # Criar DataFrame
        colunas = ['classe'] + [f'f{i}' for i in range(1, 52)]
        df_novo = pd.DataFrame(self.dados, columns=colunas)
        
        # Combinar com dados existentes
        if os.path.exists(self.caminho_arquivo):
            df_existente = pd.read_csv(self.caminho_arquivo)
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo
        
        # Salvar CSV
        df_final.to_csv(self.caminho_arquivo, index=False)
        
        # Salvar metadados
        metadados = {
            'total_amostras': len(df_final),
            'ultima_atualizacao': datetime.now().isoformat(),
            'classes_coletadas': df_final['classe'].unique().tolist(),
            'distribuicao': df_final['classe'].value_counts().to_dict()
        }
        
        with open(os.path.join(self.pasta_dados, 'metadados.json'), 'w') as f:
            json.dump(metadados, f, indent=2)
        
        # Estatísticas
        print("\n💾 DADOS SALVOS COM SUCESSO!")
        print("=" * 50)
        print(f"📁 Arquivo: {self.caminho_arquivo}")
        print(f"📊 Novas amostras: {len(self.dados)}")
        print(f"📊 Total no dataset: {len(df_final)}")
        
        # Distribuição
        print("📈 Distribuição por classe:")
        for classe in sorted(df_final['classe'].unique()):
            count = len(df_final[df_final['classe'] == classe])
            print(f"   {classe}: {count} amostras")
        
        self.total_amostras = len(df_final)
        return True
    
    def coletar(self):
        """Loop principal de coleta"""
        print("\n🎯 INICIANDO COLETA AUTOMÁTICA")
        print("=" * 50)
        print("CONTROLES:")
        print("  A : Próxima classe")
        print("  Z : Classe anterior") 
        print("  ESPAÇO : Pausar/Continuar coleta automática")
        print("  S : Salvar e continuar")
        print("  R : Resetar contador da sessão")
        print("  Q : Sair e salvar")
        print("=" * 50)
        
        # Carregar estatísticas
        self.carregar_estatisticas()
        
        # Detectar webcam
        camera_index = self.detectar_webcam_usb()
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ ERRO: Câmera não encontrada!")
            return
        
        # Índice da classe atual
        classe_index = 0
        self.classe_atual = self.classes_suportadas[classe_index]
        self.coletando = True
        
        print(f"🎯 Classe inicial: {self.classe_atual}")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Detecção de mãos
            mao_detectada = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    mao_detectada = True
                    
                    # Coleta automática
                    if self.coletando and mao_detectada:
                        tempo_atual = time.time()
                        if tempo_atual - self.ultima_coleta >= self.cooldown:
                            features = self.extrair_features(hand_landmarks)
                            if features and len(features) == 51:
                                self.dados.append([self.classe_atual] + features)
                                self.contador += 1
                                self.ultima_coleta = tempo_atual
                                print(f"✓ {self.classe_atual} - Amostra {self.contador}")
            
            # Interface
            self.mostrar_estatisticas(frame)
            cv2.imshow("Coletor LIBRAS - TraduLibras", frame)
            
            # Controles de teclado - CORRIGIDO
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # ESPAÇO - pausar/continuar
                self.coletando = not self.coletando
                status = "ATIVADA" if self.coletando else "PAUSADA"
                print(f"⏸️  Coleta {status}")
            
            elif key == ord('a'):  # A - próxima classe
                classe_index = (classe_index + 1) % len(self.classes_suportadas)
                self.classe_atual = self.classes_suportadas[classe_index]
                print(f"➡️  Classe: {self.classe_atual}")
            
            elif key == ord('z'):  # Z - classe anterior
                classe_index = (classe_index - 1) % len(self.classes_suportadas)
                self.classe_atual = self.classes_suportadas[classe_index]
                print(f"⬅️  Classe: {self.classe_atual}")
            
            elif key == ord('r'):  # Resetar sessão
                self.contador = 0
                print("🔄 Contador da sessão resetado")
            
            elif key == ord('s'):  # Salvar e continuar
                if self.dados:
                    self.salvar_dados()
                    print("💾 Dados salvos, continuando coleta...")
                else:
                    print("❌ Nenhum dado novo para salvar")
            
            elif key == ord('q'):  # Sair
                break
        
        # Finalizar
        cap.release()
        cv2.destroyAllWindows()
        
        # Salvar dados finais
        if self.dados:
            self.salvar_dados()
            print("\n🎉 COLETA FINALIZADA!")
            print("💡 Agora execute: python treinador_libras.py")
        else:
            print("\n👋 Coleta cancelada - nenhum dado salvo")

def main():
    coletor = ColetorLIBRAS()
    coletor.coletar()

if __name__ == "__main__":
    main()