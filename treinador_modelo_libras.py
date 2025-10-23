#!/usr/bin/env python3
"""
Treinador de Modelo LIBRAS
Este script treina modelos de Machine Learning com os dados coletados
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import glob

class TreinadorLIBRAS:
    def __init__(self):
        """Inicializar treinador"""
        self.model = None
        self.scaler = None
        self.features = None
        self.labels = None
        
    def carregar_dados(self, arquivo_csv):
        """Carregar dados do arquivo CSV"""
        print(f"📁 Carregando dados de: {arquivo_csv}")
        
        if not os.path.exists(arquivo_csv):
            print(f"❌ ERRO: Arquivo {arquivo_csv} não encontrado!")
            return False
        
        try:
            # Carregar CSV
            df = pd.read_csv(arquivo_csv)
            
            # Verificar estrutura
            if len(df.columns) != 52:  # 1 coluna classe + 51 features
                print(f"❌ ERRO: Formato incorreto. Esperado 52 colunas, encontrado {len(df.columns)}")
                return False
            
            # Separar features e labels
            self.labels = df.iloc[:, 0].values  # Primeira coluna é a classe
            self.features = df.iloc[:, 1:].values  # Resto são features
            
            print(f"✅ Dados carregados:")
            print(f"   - Total de amostras: {len(df)}")
            print(f"   - Features por amostra: {self.features.shape[1]}")
            print(f"   - Classes: {sorted(set(self.labels))}")
            
            # Estatísticas por classe
            for classe in sorted(set(self.labels)):
                count = list(self.labels).count(classe)
                print(f"   - {classe}: {count} amostras")
            
            return True
            
        except Exception as e:
            print(f"❌ ERRO ao carregar dados: {e}")
            return False
    
    def preparar_dados(self):
        """Preparar dados para treinamento"""
        print("\n🔧 Preparando dados para treinamento...")
        
        # Verificar se existem dados suficientes
        if len(set(self.labels)) < 2:
            print("❌ ERRO: Necessário pelo menos 2 classes diferentes!")
            return False
        
        # Separar treino e teste (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.labels
        )
        
        # Normalizar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Dados preparados:")
        print(f"   - Treino: {len(X_train_scaled)} amostras")
        print(f"   - Teste: {len(X_test_scaled)} amostras")
        print(f"   - Features normalizadas: {X_train_scaled.shape[1]}")
        
        return (X_train_scaled, X_test_scaled, y_train, y_test)
    
    def treinar_modelo(self, X_train, X_test, y_train, y_test):
        """Treinar modelo de Machine Learning"""
        print("\n🤖 Treinando modelo...")
        
        # Criar modelo ensemble (Random Forest como principal)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test)
        
        print("\n📊 RESULTADOS DO TREINAMENTO:")
        print("=" * 50)
        
        # Acurácia geral
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Acurácia: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Relatório detalhado
        print("\n📋 Relatório de Classificação:")
        report = classification_report(y_test, y_pred, output_dict=False)
        print(report)
        
        # Validação cruzada
        print("\n🔄 Validação Cruzada (5-fold):")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"🎯 CV Acurácia: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"   Scores: {[f'{c:.3f}' for c in cv_scores]}")
        
        return accuracy, cv_mean
    
    def salvar_modelo(self, precisao):
        """Salvar modelo treinado"""
        print("\n💾 Salvando modelo...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Nomes dos arquivos
        modelo_file = f'modelos/modelo_libras_{timestamp}.pkl'
        scaler_file = f'modelos/scaler_libras_{timestamp}.pkl'
        info_file = f'modelos/modelo_info_libras_{timestamp}.pkl'
        
        # Criar diretório modelos se não existir
        if not os.path.exists('modelos'):
            os.makedirs('modelos')
        
        # Informações do modelo
        model_info = {
            'timestamp': timestamp,
            'classes': sorted(set(self.labels)),
            'features_count': self.features.shape[1],
            'total_samples': len(self.features),
            'accuracy': precisao,
            'model_type': 'RandomForest',
            'creation_date': datetime.now().isoformat()
        }
        
        # Salvar modelo
        with open(modelo_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Salvar scaler
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Salvar info
        with open(info_file, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ Modelo salvo:")
        print(f"   📄 Modelo: {modelo_file}")
        print(f"   📏 Scaler: {scaler_file}")
        print(f"   ℹ️ Info: {info_file}")
        
        return modelo_file, scaler_file, info_file

def encontrar_arquivo_csv():
    """Encontrar arquivo CSV mais recente"""
    # Procurar arquivos CSV
    csv_files = glob.glob('dados_coletados/*.csv') + glob.glob('*.csv')
    
    if not csv_files:
        print("❌ ERRO: Nenhum arquivo CSV encontrado!")
        print("Execute primeiro: python coletor_dados_libras.py")
        return None
    
    # Pegar o mais recente
    arquivo_recente = max(csv_files, key=os.path.getctime)
    print(f"📁 Arquivo encontrado: {arquivo_recente}")
    
    return arquivo_recente

def main():
    """Função principal"""
    print("🚀 TREINADOR DE MODELO LIBRAS")
    print("=" * 50)
    
    # Encontrar arquivo de dados
    arquivo_csv = encontrar_arquivo_csv()
    if not arquivo_csv:
        return
    
    # Inicializar treinador
    treinador = TreinadorLIBRAS()
    
    # Carregar dados
    if not treinador.carregar_dados(arquivo_csv):
        return
    
    # Preparar dados
    dados_processados = treinador.preparar_dados()
    if not dados_processados:
        return
    
    X_train, X_test, y_train, y_test = dados_processados
    
    # Treinar modelo
    accuracy, cv_score = treinador.treinar_modelo(X_train, X_test, y_train, y_test)
    
    # Salvar modelo se precisão for adequada
    if accuracy > 0.7:  # Mínimo 70% de precisão
        treinador.salvar_modelo(accuracy)
        print("\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
    else:
        print("\n⚠️ Acurácia muito baixa! Considere:")
        print("   - Coletar mais dados")
        print("   - Melhorar qualidade dos gestos")
        print("   - Verificar variação nas amostras")
    
    print("\n👋 Programa finalizado")

if __name__ == "__main__":
    main()
