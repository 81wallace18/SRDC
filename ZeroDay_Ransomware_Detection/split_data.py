import pandas as pd
from sklearn.utils import shuffle

# 1. Carregar seu dataset completo
# Ajuste o caminho conforme necessário
df = pd.read_csv('/home/hugo_martins/SRDC/Feature_Internal_Semantic_Processing/after_feature_internal_semantic_process_data.csv')

# Mapeamento baseado no seu código main.py
# Goodware: 0
# Seen (Treino): 1, 2, 3, 4, 5, 6, 7
# Unseen (Teste): 8, 9, 10, 11

seen_families = [1, 2, 3, 4, 5, 6, 7] 
unseen_families = [8, 9, 10, 11]

# 2. Separar Goodware e Ransomware
df_goodware = df[df['family'] == 0]
df_seen_ransom = df[df['family'].isin(seen_families)]
df_unseen_ransom = df[df['family'].isin(unseen_families)]

# 3. Configurar Quantidades conforme o texto (Section 5.4)
# O texto pede:
# Treino: 448 seen ransomware + 808 goodware
# Teste: 134 unseen ransomware + 134 goodware

# Vamos tentar pegar amostras exatas, ou tudo se você tiver menos dados
# Goodware para treino
train_good = df_goodware.sample(n=min(808, len(df_goodware)), random_state=42)
# O restante do goodware fica disponível para teste, mas precisamos garantir que não haja vazamento
df_goodware_remaining = df_goodware.drop(train_good.index)
test_good = df_goodware_remaining.sample(n=min(134, len(df_goodware_remaining)), random_state=42)

# Ransomware
train_ransom = df_seen_ransom.sample(n=min(448, len(df_seen_ransom)), random_state=42)
test_ransom = df_unseen_ransom.sample(n=min(134, len(df_unseen_ransom)), random_state=42)

# 4. Concatenar e Salvar
train_final = pd.concat([train_good, train_ransom])
test_final = pd.concat([test_good, test_ransom])

# Embaralhar para o treino não ficar ordenado
train_final = shuffle(train_final, random_state=42)
test_final = shuffle(test_final, random_state=42)

train_final.to_csv('train.csv', index=False)
test_final.to_csv('test.csv', index=False)

# ... (seu código acima) ...

print("\n--- Validação Final ---")
print("Famílias presentes no TREINO (Deve ser 0 a 7):")
print(train_final['family'].unique())

print("\nFamílias presentes no TESTE (Deve ser 0, 8, 9, 10, 11):")
print(test_final['family'].unique())

print(f"Treino gerado: {len(train_final)} amostras.")
print(f"Teste (Zero-Day) gerado: {len(test_final)} amostras.")