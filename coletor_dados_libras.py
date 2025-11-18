#!/usr/bin/env python3
"""
Coletor de Dados LIBRAS
Versão com opção de CONTINUAR ou RECOMEÇAR
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

        # Verificar se existem dados anteriores
        self.dados_existentes = pd.DataFrame()
        self.classes_coletadas_anteriormente = set()
        
        if os.path.exists(self.caminho_arquivo):
            try:
                self.dados_existentes = pd.read_csv(self.caminho_arquivo)
                self.classes_coletadas_anteriormente = set(self.dados_existentes['gesture_type'].unique())
                print(f"📂 Dados anteriores encontrados ({len(self.dados_existentes)} amostras)")
                print(f"🎯 Classes já coletadas: {', '.join(sorted(self.classes_coletadas_anteriormente))}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar dados existentes: {e}")

    def perguntar_modo(self):
        """Perguntar se quer continuar ou recomeçar"""
        print("\n" + "=" * 60)
        print("🎯 MODO DE COLETA")
        print("=" * 60)
        
        if self.dados_existentes.empty:
            print("📝 Nenhum dado anterior encontrado. Iniciando nova coleta...")
            return "novo"
        
        print("1 - CONTINUAR de onde parou")
        print("2 - RECOMEÇAR do zero (apaga tudo)")
        print("3 - ESCOLHER classe específica para continuar")
        
        while True:
            try:
                opcao = input("\nEscolha uma opção (1/2/3): ").strip()
                if opcao == "1":
                    return "continuar"
                elif opcao == "2":
                    confirmacao = input("⚠️  Tem certeza? Isso apagará TODOS os dados anteriores! (s/n): ")
                    if confirmacao.lower() == 's':
                        return "recomecar"
                    else:
                        continue
                elif opcao == "3":
                    return "escolher"
                else:
                    print("❌ Opção inválida. Digite 1, 2 ou 3.")
            except KeyboardInterrupt:
                print("\n👋 Saindo...")
                exit()

    def escolher_classe_inicial(self, classes):
        """Permitir escolher de qual classe começar"""
        print("\n" + "=" * 60)
        print("🎯 ESCOLHER CLASSE INICIAL")
        print("=" * 60)
        
        for i, classe in enumerate(classes):
            status = "✅" if (classe[1] if isinstance(classe, tuple) else classe) in self.classes_coletadas_anteriormente else "❌"
            display_name = classe[1] if isinstance(classe, tuple) else classe
            print(f"{i+1:2d}. {status} {display_name}")
        
        while True:
            try:
                escolha = input(f"\nDigite o número da classe para começar (1-{len(classes)}): ").strip()
                indice = int(escolha) - 1
                if 0 <= indice < len(classes):
                    return indice
                else:
                    print(f"❌ Número inválido. Digite entre 1 e {len(classes)}")
            except ValueError:
                print("❌ Digite um número válido.")
            except KeyboardInterrupt:
                print("\n👋 Saindo...")
                exit()

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

    def mostrar_status(self, frame, classe, contador, indice_atual, total_classes, modo):
        """Mostrar status na tela"""
        classe_display = classe[1] if isinstance(classe, tuple) else classe
        cv2.putText(frame, f"CLASSE: {classe_display} ({indice_atual+1}/{total_classes})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AMOSTRAS (sessão): {contador}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"MODO: {modo}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE: Próxima classe | ESC: Sair e Salvar", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def coletar_dados(self):
        """Função principal"""
        # Definir todas as classes
        classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [(" ", "ESPACO"), (".", "PONTO")]
        total_classes = len(classes)
        
        # Perguntar modo de operação
        modo = self.perguntar_modo()
        
        # Definir classe inicial baseada no modo
        if modo == "recomecar":
            indice_classe = 0
            self.dados_existentes = pd.DataFrame()  # Limpar dados anteriores
            self.classes_coletadas_anteriormente = set()
            print("🆕 Iniciando coleta DO ZERO...")
        elif modo == "continuar":
            # Encontrar primeira classe não coletada
            indice_classe = 0
            for i, classe in enumerate(classes):
                display_name = classe[1] if isinstance(classe, tuple) else classe
                if display_name not in self.classes_coletadas_anteriormente:
                    indice_classe = i
                    break
            print(f"➡️ Continuando da classe: {classes[indice_classe][1] if isinstance(classes[indice_classe], tuple) else classes[indice_classe]}")
        elif modo == "escolher":
            indice_classe = self.escolher_classe_inicial(classes)
            print(f"🎯 Iniciando da classe escolhida: {classes[indice_classe][1] if isinstance(classes[indice_classe], tuple) else classes[indice_classe]}")
        else:  # modo novo
            indice_classe = 0
        
        self.classe_atual = classes[indice_classe]
        
        print("=" * 60)
        print("COLETOR DE DADOS LIBRAS - MODO FLEXÍVEL")
        print("=" * 60)
        print("Instruções:")
        print("👉 Mostre o gesto da classe atual na frente da câmera")
        print("👉 Pressione ESPAÇO para mudar de classe")
        print("👉 Pressione ESC para salvar e sair")
        print("=" * 60)

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
                            classe_para_salvar = self.classe_atual[1] if isinstance(self.classe_atual, tuple) else self.classe_atual
                            self.dados_coletados.append([classe_para_salvar] + pontos)
                            self.contador_amostras += 1
                            ultima_detecacao = time.time()
                            print(f"📝 Amostra {self.contador_amostras} coletada ({classe_para_salvar})")

            modo_display = {
                "novo": "NOVA COLETA",
                "continuar": "CONTINUANDO", 
                "recomecar": "RECOMEÇANDO",
                "escolher": "CLASSE ESCOLHIDA"
            }.get(modo, "COLETANDO")
            
            self.mostrar_status(frame, self.classe_atual, self.contador_amostras, indice_classe, total_classes, modo_display)
            cv2.imshow("Coletor LIBRAS", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:  # ESC - Salvar e sair
                print("\n💾 Salvando e saindo...")
                break
            elif tecla == 32:  # SPACE - Próxima classe
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
        """Salvar os dados coletados (mantendo ou substituindo anteriores conforme o modo)"""
        if not self.dados_coletados:
            print("❌ Nenhum dado coletado nesta sessão.")
            return

        print("\n💾 Salvando dados...")

        # Formatar colunas
        columns = ['gesture_type'] + [f'feature_{i+1}' for i in range(51)]
        novos_dados = pd.DataFrame(self.dados_coletados, columns=columns)

        # Se existem dados anteriores e não estamos no modo recomeçar, unir tudo
        if not self.dados_existentes.empty and len(self.dados_existentes) > 0:
            # Remover duplicatas da mesma classe (evita dados antigos da mesma classe)
            classes_coletadas_agora = set(novos_dados['gesture_type'])
            dados_anteriores_filtrados = self.dados_existentes[~self.dados_existentes['gesture_type'].isin(classes_coletadas_agora)]
            
            df_final = pd.concat([dados_anteriores_filtrados, novos_dados], ignore_index=True)
            print(f"🔄 Mantendo dados anteriores de outras classes ({len(dados_anteriores_filtrados)} amostras)")
        else:
            df_final = novos_dados

        df_final.to_csv(self.caminho_arquivo, index=False)
        total_amostras = len(df_final)
        classes_unicas = df_final['gesture_type'].unique()

        print("=" * 60)
        print("✅ DADOS SALVOS COM SUCESSO!")
        print(f"📁 Arquivo: {self.caminho_arquivo}")
        print(f"📊 Total de amostras: {total_amostras}")
        print(f"🎯 Classes coletadas: {', '.join(sorted(classes_unicas))}")
        print("=" * 60)


def main():
    print("🚀 Iniciando Coletor LIBRAS (Modo Flexível)...")
    coletor = ColetorLIBRAS()
    coletor.coletar_dados()
    print("👋 Finalizado!")


if __name__ == "__main__":
    main()