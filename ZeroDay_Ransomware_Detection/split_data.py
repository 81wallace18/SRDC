import pandas as pd

# Carrega seu CSV arrumado
df = pd.read_csv('/home/hugo_martins/Project_Hugo/SRDC/after_feature_internal_semantic_process_data.csv', sep=',')

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
df_goodware_test = df[df['family'] == 0].sample(n=100) # Pega 100 goodwares aleatórios
df_test = pd.concat([df_zero_day, df_goodware_test])

# Salva os arquivos
df_train.to_csv('train_zeroday.csv', index=False)
df_test.to_csv('test_zeroday.csv', index=False)

print("Arquivos criados: train_zeroday.csv e test_zeroday.csv")