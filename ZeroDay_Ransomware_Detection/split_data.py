import pandas as pd
import numpy as np

# Carrega seu CSV arrumado
df = pd.read_csv('/home/hugo/Projects/SRDC/after_feature_internal_semantic_process_data.csv', sep=',')

print(f"Dataset original: {df.shape[0]} amostras")

# Função para criar texto para BERT a partir das features existentes
def create_bert_text(row):
    """Combina as features existentes em texto para BERT"""
    text_parts = []

    # Adiciona API Features se existir e não for vazia
    if 'apiFeatures' in row and pd.notna(row['apiFeatures']) and str(row['apiFeatures']).strip():
        text_parts.append(str(row['apiFeatures']))

    # Adiciona Registry Features se existir e não for vazia
    if 'regFeatures' in row and pd.notna(row['regFeatures']) and str(row['regFeatures']).strip():
        text_parts.append(str(row['regFeatures']))

    # Adiciona File Features se existir e não for vazia
    if 'filesFeatures' in row and pd.notna(row['filesFeatures']) and str(row['filesFeatures']).strip():
        text_parts.append(str(row['filesFeatures']))

    # Adiciona String Features se existir e não for vazia
    if 'strFeatures' in row and pd.notna(row['strFeatures']) and str(row['strFeatures']).strip():
        text_parts.append(str(row['strFeatures']))

    # Retorna o texto combinado ou texto padrão se não houver features
    return " ".join(text_parts) if text_parts else "empty features"

# Adiciona coluna bert_text no final para manter formatação original
df['bert_text'] = df.apply(create_bert_text, axis=1)

print(f"Texto BERT criado para todas as {df.shape[0]} amostras")

# LÓGICA DE ZERO DAY:
# Vamos treinar com todas as famílias, EXCETO a família 11 (exemplo)
# A família 11 será a "desconhecida" (Zero Day)

# 1. Cria o Dataset de TREINO (Famílias conhecidas + Goodware)
# Exclui a família 11
df_train = df[df['family'] != 11]

# 2. Cria o Dataset de TESTE (Apenas a família 11 + Alguns Goodwares para controle)
# Pega apenas a família 11
df_zero_day = df[df['family'] == 11]
# Pega alguns goodwares (família 0) para o teste não ser só vírus
# Limita ao número disponível de goodwares
num_goodware = min(50, len(df[df['family'] == 0]))
df_goodware_test = df[df['family'] == 0].sample(n=num_goodware, random_state=42)
df_test = pd.concat([df_zero_day, df_goodware_test])

# Salva os arquivos mantendo a formatação original + a nova coluna bert_text
df_train.to_csv('train_zeroday.csv', index=False)
df_test.to_csv('test_zeroday.csv', index=False)

print(f"\nArquivos criados:")
print(f"✓ train_zeroday.csv: {len(df_train)} amostras (todas as famílias exceto a 11)")
print(f"✓ test_zeroday.csv: {len(df_test)} amostras ({len(df_zero_day)} Zero Day + {len(df_goodware_test)} Goodware)")
print(f"\nColuna 'bert_text' adicionada no final mantendo formatação original")