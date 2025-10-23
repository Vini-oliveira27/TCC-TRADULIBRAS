#!/usr/bin/env python3
"""
Coletor de Dados LIBRAS
Versão com salvamento contínuo em um único CSV
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

class ColetorLIBRAS:
    def __init__(self, pasta_dados='dados_coletados', arquivo_csv='gestos_libras.csv'):
        """Inicializar coletor"""
        self.pasta_dados = pasta_dados
        self.arquivo_csv = arquivo_csv
        self.caminho_arquivo = os.path.join(self.pasta_dados, self.arquivo_csv)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        if not os.path.exists(pasta_dados):
            os.makedirs(pasta_dados)
            
        self.dados_coletados = []
        self.classe_atual = None
        self.contador_amostras = 0

        # Carregar dados anteriores (se existirem)
        if os.path.exists(self.caminho_arquivo):
            try:
                self.dados_existentes = pd.read_csv(self.caminho_arquivo)
                print(f"📂 Dados anteriores carregados ({len(self.dados_existentes)} amostras)")
            except Exception as e:
                print(f"⚠️ Erro ao carregar dados existentes: {e}")
                self.dados_existentes = pd.DataFrame()
        else:
            self.dados_existentes = pd.DataFrame()

    def processar_landmarks(self, hand_landmarks):
        """Processar landmarks da mão"""
        if not hand_landmarks:
            return None

        wrist = hand_landmarks.landmark[0]
        features = []

        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y
            ])

        # Features extras
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        features.extend([
            abs(thumb_tip.x - wrist.x) + abs(thumb_tip.y - wrist.y),
            abs(index_tip.x - wrist.x) + abs(index_tip.y - wrist.y),
            abs(middle_tip.x - wrist.x) + abs(middle_tip.y - wrist.y),
            abs(ring_tip.x - wrist.x) + abs(ring_tip.y - wrist.y),
            abs(pinky_tip.x - wrist.x) + abs(pinky_tip.y - wrist.y)
        ])

        features.extend([
            abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y),
            abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y),
            abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y),
            abs(ring_tip.x - pinky_tip.x) + abs(ring_tip.y - pinky_tip.y)
        ])

        return features  # total: 51 features

    def mostrar_status(self, frame, classe, contador, indice_atual, total_classes):
        """Mostrar status na tela"""
        # Mostrar o símbolo real se for tuple (ESPACO ou PONTO_FINAL)
        classe_display = classe[1] if isinstance(classe, tuple) else classe
        cv2.putText(frame, f"CLASSE: {classe_display} ({indice_atual+1}/{total_classes})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AMOSTRAS (sessão): {contador}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Próxima classe | ESC: Sair", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def coletar_dados(self):
        """Função principal"""
        print("=" * 60)
        print("COLETOR DE DADOS LIBRAS - SALVAMENTO CONTÍNUO")
        print("=" * 60)
        print("Instruções:")
        print("👉 Mostre o gesto da classe atual na frente da câmera")
        print("👉 Pressione ESPAÇO para mudar de classe")
        print("👉 Pressione ESC para encerrar")
        print("=" * 60)
        
        # Letras + sinais especiais mapeados para seus símbolos
        classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [(" "), (".")]
        indice_classe = 0
        self.classe_atual = classes[indice_classe]
        total_classes = len(classes)
        print(f"🆕 Iniciando coleta para classe: {self.classe_atual[1] if isinstance(self.classe_atual, tuple) else self.classe_atual}")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("❌ ERRO: Não foi possível acessar a câmera!")
            return

        cooldown_detecacao = 0.5
        ultima_detecacao = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if time.time() - ultima_detecacao >= cooldown_detecacao:
                        pontos = self.processar_landmarks(hand_landmarks)
                        if pontos:
                            # Salvar o símbolo real se for tuple
                            classe_para_salvar = self.classe_atual[1] if isinstance(self.classe_atual, tuple) else self.classe_atual
                            self.dados_coletados.append([classe_para_salvar] + pontos)
                            self.contador_amostras += 1
                            ultima_detecacao = time.time()
                            print(f"📝 Amostra {self.contador_amostras} coletada ({classe_para_salvar})")

            self.mostrar_status(frame, self.classe_atual, self.contador_amostras, indice_classe, total_classes)
            cv2.imshow("Coletor LIBRAS", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:  # ESC
                break
            elif tecla == 32:  # SPACE
                indice_classe += 1
                if indice_classe < len(classes):
                    self.classe_atual = classes[indice_classe]
                    self.contador_amostras = 0
                    display_name = self.classe_atual[1] if isinstance(self.classe_atual, tuple) else self.classe_atual
                    print(f"\n➡️ Mudando para classe: {display_name}")
                else:
                    print("\n✅ Todas as classes coletadas!")
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.salvar_dados()

    def salvar_dados(self):
        """Salvar os dados coletados (sem sobrescrever)"""
        if not self.dados_coletados:
            print("❌ Nenhum dado coletado nesta sessão.")
            return

        print("\n💾 Salvando dados...")

        # Formatar colunas
        columns = ['gesture_type'] + [f'feature_{i+1}' for i in range(51)]
        novos_dados = pd.DataFrame(self.dados_coletados, columns=columns)

        # Se já existirem dados antigos, unir tudo
        if not self.dados_existentes.empty:
            df_final = pd.concat([self.dados_existentes, novos_dados], ignore_index=True)
        else:
            df_final = novos_dados

        df_final.to_csv(self.caminho_arquivo, index=False)
        total_amostras = len(df_final)
        classes_unicas = df_final['gesture_type'].unique()

        print("=" * 60)
        print("✅ DADOS SALVOS (modo contínuo)")
        print(f"📁 Arquivo: {self.caminho_arquivo}")
        print(f"📊 Total acumulado: {total_amostras}")
        print(f"🎯 Classes presentes: {', '.join(classes_unicas)}")
        print("=" * 60)


def main():
    print("🚀 Iniciando Coletor LIBRAS (modo contínuo)...")
    coletor = ColetorLIBRAS()
    coletor.coletar_dados()
    print("👋 Finalizado!")


if __name__ == "__main__":
    main()
