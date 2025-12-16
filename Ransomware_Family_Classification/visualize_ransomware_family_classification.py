#!/usr/bin/env python3
"""
Script de visualização para resultados da tarefa Ransomware Family Classification.
Este script lê os resultados gerados pelo classificador e cria gráficos detalhados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Mapeamento das famílias de ransomware
FAMILY_MAPPING = {
    0: 'Critroni',
    1: 'CryptLocker',
    2: 'CryptoWall',
    3: 'KOLLAH',
    4: 'Kovter',
    5: 'Locker',
    6: 'MATSNU',
    7: 'PGPCODER',
    8: 'Reveton',
    9: 'TeslaCrypt',
    10: 'Trojan-Ransom',
    11: 'Goodware'
}

class RansomwareFamilyVisualizer:
    def __init__(self, data_path='after_feature_internal_semantic_process_data.csv', results_dir='Ransomware_Family_Classification'):
        self.data_path = data_path
        self.results_dir = results_dir
        self.df_data = None
        self.load_data()

    def load_data(self):
        """Carrega os dados principais"""
        try:
            self.df_data = pd.read_csv(self.data_path)
            self.df_data.fillna('', inplace=True)
            print(f"✓ Dados carregados: {len(self.df_data)} amostras")
        except FileNotFoundError:
            print(f"❌ Arquivo de dados não encontrado: {self.data_path}")
            return

    def analyze_data_distribution(self):
        """Analisa a distribuição dos dados por família"""
        if self.df_data is None:
            return

        plt.figure(figsize=(15, 10))

        # Gráfico 1: Distribuição por família
        plt.subplot(2, 2, 1)
        family_counts = self.df_data['family'].value_counts().sort_index()
        family_labels = [FAMILY_MAPPING.get(i, f'Unknown_{i}') for i in family_counts.index]

        bars = plt.bar(family_labels, family_counts.values)
        plt.title('Distribuição de Amostras por Família de Ransomware', fontsize=14, fontweight='bold')
        plt.xlabel('Família')
        plt.ylabel('Número de Amostras')
        plt.xticks(rotation=45, ha='right')

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')

        # Gráfico 2: Pie chart da distribuição
        plt.subplot(2, 2, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(family_counts)))
        plt.pie(family_counts.values, labels=family_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Proporção de Amostras por Família', fontsize=14, fontweight='bold')

        # Gráfico 3: Análise de features por família
        plt.subplot(2, 2, 3)
        feature_counts = []
        for col in ['apiFeatures', 'regFeatures', 'filesFeatures', 'strFeatures']:
            count = self.df_data[col].apply(lambda x: 1 if x.strip() else 0).sum()
            feature_counts.append(count)

        features = ['API Calls', 'Registry', 'Files', 'Strings']
        bars = plt.bar(features, feature_counts)
        plt.title('Presença de Features por Tipo', fontsize=14, fontweight='bold')
        plt.ylabel('Número de Amostras com Feature')
        plt.xticks(rotation=45)

        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')

        # Gráfico 4: Densidade de features
        plt.subplot(2, 2, 4)
        api_lengths = self.df_data['apiFeatures'].apply(lambda x: len(x.split('.')) if x.strip() else 0)
        plt.hist(api_lengths, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribuição do Número de API Calls', fontsize=14, fontweight='bold')
        plt.xlabel('Número de API Calls')
        plt.ylabel('Frequência')

        plt.tight_layout()
        plt.savefig(f'visualizations/ransomware_family_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_training_results(self):
        """Analisa resultados de treinamento se existirem"""
        # Procurar por arquivos de resultados
        result_files = glob.glob('result*.txt') + glob.glob(f'{self.results_dir}/result*.txt')
        csv_files = glob.glob('fold_*_epoch_*.csv')

        if not result_files and not csv_files:
            print("⚠️  Nenhum arquivo de resultado encontrado. Execute o treinamento primeiro.")
            return

        # Análise de arquivos CSV (se existirem)
        if csv_files:
            self._analyze_csv_results(csv_files)

        # Análise de arquivos de texto (se existirem)
        if result_files:
            self._analyze_text_results(result_files)

    def _analyze_csv_results(self, csv_files):
        """Analisa resultados dos arquivos CSV gerados por fold"""
        all_metrics = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0)
                fold_info = self._extract_fold_info(csv_file)

                # Extrair métricas principais
                if 'accuracy' in df.index:
                    accuracy = df.loc['accuracy', 'precision']  # Ajuste conforme estrutura real
                    all_metrics.append({
                        'fold': fold_info['fold'],
                        'epoch': fold_info['epoch'],
                        'accuracy': accuracy,
                        'file': csv_file
                    })
            except Exception as e:
                print(f"Erro ao ler {csv_file}: {e}")

        if all_metrics:
            self._plot_training_progress(all_metrics)

    def _analyze_text_results(self, result_files):
        """Analisa resultados dos arquivos de texto"""
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                print(f"\n📊 Resultados de {result_file}:")
                print("="*50)
                print(content[:500] + "..." if len(content) > 500 else content)
            except Exception as e:
                print(f"Erro ao ler {result_file}: {e}")

    def _extract_fold_info(self, filename):
        """Extrai informações de fold e epoch do nome do arquivo"""
        import re
        match = re.search(r'fold_(\d+)_epoch_(\d+)', filename)
        if match:
            return {'fold': int(match.group(1)), 'epoch': int(match.group(2))}
        return {'fold': 0, 'epoch': 0}

    def _plot_training_progress(self, metrics):
        """Plota o progresso do treinamento"""
        if not metrics:
            return

        df_metrics = pd.DataFrame(metrics)

        plt.figure(figsize=(15, 5))

        # Gráfico 1: Accuracy por Fold
        plt.subplot(1, 3, 1)
        for fold in df_metrics['fold'].unique():
            fold_data = df_metrics[df_metrics['fold'] == fold]
            plt.plot(fold_data['epoch'], fold_data['accuracy'], marker='o', label=f'Fold {fold}')

        plt.title('Accuracy por Fold e Época', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfico 2: Accuracy média por época
        plt.subplot(1, 3, 2)
        mean_accuracy = df_metrics.groupby('epoch')['accuracy'].mean()
        std_accuracy = df_metrics.groupby('epoch')['accuracy'].std()

        plt.plot(mean_accuracy.index, mean_accuracy.values, marker='s', linewidth=2, markersize=8)
        plt.fill_between(mean_accuracy.index,
                        mean_accuracy.values - std_accuracy.values,
                        mean_accuracy.values + std_accuracy.values,
                        alpha=0.2)
        plt.title('Accuracy Média por Época', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Accuracy Média')
        plt.grid(True, alpha=0.3)

        # Gráfico 3: Box plot por fold
        plt.subplot(1, 3, 3)
        df_metrics.boxplot(column='accuracy', by='fold', ax=plt.gca())
        plt.title('Distribuição de Accuracy por Fold', fontsize=14, fontweight='bold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.suptitle('')  # Remove o título automático do pandas

        plt.tight_layout()
        plt.savefig(f'visualizations/ransomware_family_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_confusion_matrix_placeholder(self):
        """Gera um placeholder para matriz de confusão"""
        plt.figure(figsize=(12, 10))

        # Criar uma matriz de confusão exemplo (substituir com dados reais quando disponível)
        n_classes = len(FAMILY_MAPPING)
        confusion_matrix_example = np.eye(n_classes) + np.random.rand(n_classes, n_classes) * 0.2
        confusion_matrix_example = confusion_matrix_example / confusion_matrix_example.sum(axis=1, keepdims=True)

        family_labels = list(FAMILY_MAPPING.values())

        sns.heatmap(confusion_matrix_example,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=family_labels,
                   yticklabels=family_labels)

        plt.title('Matriz de Confusão (Exemplo)', fontsize=16, fontweight='bold')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(f'visualizations/ransomware_family_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """Gera um relatório resumido"""
        print("\n" + "="*60)
        print("📊 RELATÓRIO: RANSOMWARE FAMILY CLASSIFICATION")
        print("="*60)

        if self.df_data is not None:
            print(f"\n📋 Dataset Overview:")
            print(f"   • Total de amostras: {len(self.df_data)}")
            print(f"   • Número de famílias: {self.df_data['family'].nunique()}")
            print(f"   • Features disponíveis: {list(self.df_data.columns[1:])}")

            print(f"\n👥 Distribuição por Família:")
            family_counts = self.df_data['family'].value_counts().sort_index()
            for family_id, count in family_counts.items():
                family_name = FAMILY_MAPPING.get(family_id, f'Unknown_{family_id}')
                percentage = (count / len(self.df_data)) * 100
                print(f"   • {family_name}: {count} amostras ({percentage:.1f}%)")

        print(f"\n📈 Métricas de Performance:")
        print("   • Aguardando resultados do treinamento para exibir métricas...")

        print(f"\n💡 Recomendações:")
        print("   1. Execute o treinamento com: python ransomware_family_classifier.py")
        print("   2. Verifique os arquivos result.txt e fold_*_epoch_*.csv")
        print("   3. Execute este script novamente para visualizar os resultados")

        print("\n" + "="*60)

def main():
    """Função principal"""
    print("🔍 Iniciando Visualização: Ransomware Family Classification")
    print("="*60)

    # Criar diretório de visualizações se não existir
    os.makedirs('visualizations', exist_ok=True)

    # Inicializar visualizador
    visualizer = RansomwareFamilyVisualizer()

    # Gerar visualizações
    print("\n1. Analisando distribuição dos dados...")
    visualizer.analyze_data_distribution()

    print("\n2. Analisando resultados de treinamento...")
    visualizer.analyze_training_results()

    print("\n3. Gerando matriz de confusão (exemplo)...")
    visualizer.generate_confusion_matrix_placeholder()

    print("\n4. Gerando relatório resumido...")
    visualizer.generate_summary_report()

    print(f"\n✅ Visualizações salvas em: visualizations/")
    print("   • ransomware_family_data_distribution.png")
    print("   • ransomware_family_training_progress.png")
    print("   • ransomware_family_confusion_matrix.png")

if __name__ == "__main__":
    main()