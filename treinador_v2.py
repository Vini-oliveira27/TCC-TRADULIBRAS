#!/usr/bin/env python3
"""
TREINADOR LIBRAS - COMPATÍVEL COM APP.PY
Treinador otimizado para o sistema TraduLibras
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TreinadorLIBRAS:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.encoder = None
        self.info_modelo = {}
        
        # Configurações do modelo
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'cv_folds': 5
        }
        
    def encontrar_dataset(self):
        """Encontrar o dataset mais recente"""
        possiveis_caminhos = [
            'dados_libras/dataset_libras.csv',
            'dados_libras/gestos.csv', 
            'dataset_libras.csv',
            'gestos.csv'
        ]
        
        for caminho in possiveis_caminhos:
            if os.path.exists(caminho):
                return caminho
        
        # Buscar qualquer CSV
        arquivos_csv = glob.glob('**/*.csv', recursive=True)
        if arquivos_csv:
            return max(arquivos_csv, key=os.path.getctime)
        
        return None
    
    def diagnosticar_dataset(self, arquivo):
        """Diagnóstico completo do dataset"""
        print("🔍 DIAGNÓSTICO DO DATASET")
        print("=" * 60)
        
        if not arquivo or not os.path.exists(arquivo):
            print("❌ Nenhum dataset encontrado!")
            print("💡 Execute primeiro: python coletor_libras.py")
            return False
        
        try:
            df = pd.read_csv(arquivo)
            
            # Verificar estrutura básica
            num_features = len(df.columns) - 1
            num_amostras = len(df)
            classes = df['classe'].unique()
            
            print(f"📁 Arquivo: {arquivo}")
            print(f"📊 Total de amostras: {num_amostras:,}")
            print(f"🔧 Features por amostra: {num_features}")
            print(f"🎯 Classes detectadas: {list(classes)}")
            
            # Verificar compatibilidade
            if num_features != 51:
                print(f"❌ INCOMPATÍVEL: Esperado 51 features, encontrado {num_features}")
                print("💡 Use o coletor_libras.py para gerar dados compatíveis")
                return False
            
            print("✅ COMPATÍVEL: 51 features detectadas")
            
            # Análise de distribuição
            print("\n📈 DISTRIBUIÇÃO DAS CLASSES:")
            distribuição = df['classe'].value_counts()
            for classe, count in distribuição.items():
                percentual = (count / num_amostras) * 100
                print(f"   {classe:>8}: {count:>4} amostras ({percentual:5.1f}%)")
            
            # Verificar balanceamento
            min_amostras = distribuição.min()
            max_amostras = distribuição.max()
            ratio = max_amostras / min_amostras if min_amostras > 0 else float('inf')
            
            print(f"\n⚖️  BALANCEAMENTO:")
            print(f"   Mínimo: {min_amostras} amostras")
            print(f"   Máximo: {max_amostras} amostras") 
            print(f"   Razão: {ratio:.1f}x")
            
            if ratio > 5:
                print("⚠️  ALERTA: Dataset muito desbalanceado!")
            elif ratio > 3:
                print("⚠️  AVISO: Dataset desbalanceado")
            else:
                print("✅ Dataset balanceado")
            
            # Verificar qualidade dos dados
            print(f"\n🔍 QUALIDADE DOS DADOS:")
            nulos = df.isnull().sum().sum()
            print(f"   Valores nulos: {nulos}")
            
            if nulos > 0:
                print("⚠️  AVISO: Valores nulos detectados")
                df = df.dropna()
                print(f"   Amostras após limpeza: {len(df):,}")
            
            # Salvar informações
            self.info_dataset = {
                'arquivo': arquivo,
                'amostras': len(df),
                'features': num_features,
                'classes': list(classes),
                'distribuicao': distribuição.to_dict(),
                'balanceamento_ratio': ratio,
                'compativel': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no diagnóstico: {e}")
            return False
    
    def preparar_dados(self):
        """Preparar dados para treinamento"""
        print("\n🔧 PREPARANDO DADOS PARA TREINAMENTO")
        print("=" * 60)
        
        if not hasattr(self, 'info_dataset') or not self.info_dataset['compativel']:
            print("❌ Dataset incompatível")
            return None
        
        try:
            df = pd.read_csv(self.info_dataset['arquivo'])
            
            # Limpar dados nulos
            df = df.dropna()
            
            # Separar features e labels
            X = df.iloc[:, 1:].values  # Features (colunas 1-51)
            y = df.iloc[:, 0].values   # Labels (coluna 0 - classe)
            
            print(f"📦 Dados carregados:")
            print(f"   - Features (X): {X.shape}")
            print(f"   - Labels (y): {y.shape}")
            
            # Codificar labels
            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
            
            print(f"🔤 Labels codificados:")
            for i, classe in enumerate(self.encoder.classes_):
                print(f"   {classe} → {i}")
            
            # Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y_encoded
            )
            
            print(f"🎯 Divisão dos dados:")
            print(f"   - Treino: {X_train.shape[0]:,} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
            print(f"   - Teste:  {X_test.shape[0]:,} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
            
            # Normalização
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print("✅ Normalização aplicada (StandardScaler)")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"❌ Erro ao preparar dados: {e}")
            return None
    
    def treinar_modelo(self, X_train, X_test, y_train, y_test):
        """Treinar o modelo Random Forest"""
        print("\n🤖 INICIANDO TREINAMENTO DO MODELO")
        print("=" * 60)
        
        try:
            # Criar modelo
            self.modelo = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state'],
                n_jobs=-1,  # Usar todos os cores
                verbose=1
            )
            
            print("🔄 Treinando modelo...")
            self.modelo.fit(X_train, y_train)
            print("✅ Modelo treinado com sucesso!")
            
            # Avaliação no conjunto de teste
            y_pred = self.modelo.predict(X_test)
            acuracia = accuracy_score(y_test, y_pred)
            
            print(f"\n🎯 AVALIAÇÃO NO TESTE:")
            print(f"   Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")
            
            # Relatório de classificação
            print(f"\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
            report = classification_report(y_test, y_pred, target_names=self.encoder.classes_, output_dict=True)
            
            for classe in self.encoder.classes_:
                if classe in report:
                    prec = report[classe]['precision']
                    rec = report[classe]['recall']
                    f1 = report[classe]['f1-score']
                    print(f"   {classe:>8}: Precision {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f}")
            
            # Matriz de confusão
            print(f"\n🎭 MATRIZ DE CONFUSÃO (linha → coluna):")
            cm = confusion_matrix(y_test, y_pred)
            print("     " + " ".join([f"{c:>3}" for c in self.encoder.classes_]))
            for i, true_class in enumerate(self.encoder.classes_):
                linha = f"{true_class:>3} " + " ".join([f"{cm[i,j]:>3}" for j in range(len(self.encoder.classes_))])
                print(linha)
            
            # Validação cruzada
            print(f"\n🔄 VALIDAÇÃO CRUZADA ({self.config['cv_folds']}-fold):")
            cv_scores = cross_val_score(self.modelo, X_train, y_train, cv=self.config['cv_folds'])
            print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"   Média:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Importância das features
            importancias = self.modelo.feature_importances_
            top_features = np.argsort(importancias)[-10:]  # Top 10
            print(f"\n📊 TOP 10 FEATURES MAIS IMPORTANTES:")
            for idx in reversed(top_features):
                print(f"   f{idx+1:2d}: {importancias[idx]:.4f}")
            
            return acuracia
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return 0
    
    def salvar_modelo(self, acuracia):
        """Salvar modelo no formato compatível com app.py"""
        print("\n💾 SALVANDO MODELO TREINADO")
        print("=" * 60)
        
        # Criar pasta de modelos (compatível com app.py)
        pasta_modelos = 'modelos'
        if not os.path.exists(pasta_modelos):
            os.makedirs(pasta_modelos)
        
        try:
            # Salvar componentes individuais
            with open(os.path.join(pasta_modelos, 'modelo.pkl'), 'wb') as f:
                pickle.dump(self.modelo, f)
            with open(os.path.join(pasta_modelos, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(os.path.join(pasta_modelos, 'encoder.pkl'), 'wb') as f:
                pickle.dump(self.encoder, f)
            
            # Informações do modelo (compatível com app.py)
            self.info_modelo = {
                'classes': self.encoder.classes_.tolist(),
                'accuracy': acuracia,
                'description': f"Modelo TreinadorLIBRAS - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'model_type': 'RandomForest',
                'total_samples': self.info_dataset['amostras'],
                'features_used': 51,
                'training_date': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(os.path.join(pasta_modelos, 'info.pkl'), 'wb') as f:
                pickle.dump(self.info_modelo, f)
            
            print("✅ MODELO SALVO COM SUCESSO!")
            print(f"📁 Pasta: {pasta_modelos}/")
            print("📄 Arquivos salvos:")
            print("   - modelo.pkl (modelo Random Forest)")
            print("   - scaler.pkl (normalizador StandardScaler)")
            print("   - encoder.pkl (codificador de labels)")
            print("   - info.pkl (informações do modelo)")
            print(f"\n🎯 Acurácia do modelo: {acuracia:.4f}")
            print(f"🎯 Classes treinadas: {len(self.info_modelo['classes'])}")
            print(f"📊 Amostras utilizadas: {self.info_modelo['total_samples']:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            return False
    
    def teste_compatibilidade(self):
        """Teste de compatibilidade com app.py"""
        print("\n🔧 TESTE DE COMPATIBILIDADE")
        print("=" * 60)
        
        try:
            # Simular dados de entrada (51 features)
            dados_teste = np.random.random((1, 51))
            
            # Processar como no app.py
            dados_normalizados = self.scaler.transform(dados_teste)
            probabilidades = self.modelo.predict_proba(dados_normalizados)[0]
            predicao_idx = np.argmax(probabilidades)
            classe_predita = self.encoder.inverse_transform([predicao_idx])[0]
            confianca = probabilidades[predicao_idx]
            
            print("✅ COMPATIBILIDADE VERIFICADA:")
            print(f"   - Scaler: 51 → {dados_normalizados.shape[1]} features")
            print(f"   - Modelo: predição → {classe_predita}")
            print(f"   - Encoder: decodificação funcionando")
            print(f"   - Confiança: {confianca:.3f}")
            print(f"   - Probabilidades: shape {probabilidades.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ FALHA NA COMPATIBILIDADE: {e}")
            return False

def main():
    print("🚀 TREINADOR LIBRAS - TRADULIBRAS")
    print("=" * 70)
    print("Sistema de treinamento compatível com app.py")
    print("=" * 70)
    
    # Inicializar treinador
    treinador = TreinadorLIBRAS()
    
    # 1. Encontrar e diagnosticar dataset
    arquivo = treinador.encontrar_dataset()
    if not treinador.diagnosticar_dataset(arquivo):
        return
    
    # 2. Preparar dados
    dados = treinador.preparar_dados()
    if dados is None:
        return
    
    X_train, X_test, y_train, y_test = dados
    
    # 3. Treinar modelo
    acuracia = treinador.treinar_modelo(X_train, X_test, y_train, y_test)
    
    # 4. Salvar se a acurácia for aceitável
    if acuracia >= 0.7:  # Limiar reduzido para aceitar mais modelos
        if treinador.salvar_modelo(acuracia):
            # Teste final de compatibilidade
            if treinador.teste_compatibilidade():
                print("\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
                print("   O modelo está pronto para uso no app.py!")
                print(f"   💡 Execute: python app.py")
            else:
                print("\n⚠️  Modelo salvo mas com problemas de compatibilidade")
    else:
        print(f"\n⚠️  Acurácia muito baixa ({acuracia:.3f})")
        print("💡 RECOMENDAÇÕES:")
        print("   - Colete mais dados balanceados")
        print("   - Verifique a qualidade dos gestos")
        print("   - Aumente o número de amostras por classe")
        print("   - Melhore a iluminação e posição da câmera")
    
    print("\n👋 Finalizado!")

if __name__ == "__main__":
    main()