import pandas as pd

# Charger le dataset agrégé
# Ce fichier est déjà généré dans beacon_dataset.csv

df = pd.read_csv('beacon_dataset.csv')

# Exemple de règle donnée dans le sujet
# connection_count >= 20
# ET recurrent_ratio >= 0.75
# ET delta_cv <= 0.25

df['rule_prediction'] = (
    (df['connection_count'] >= 20) &
    (df['recurrent_ratio'] >= 0.75) &
    (df['delta_cv'] <= 0.25)
).astype(int)

# Matrice de confusion
# label = 1 si malveillant, 0 si normal

tp = ((df['rule_prediction'] == 1) & (df['label'] == 1)).sum()
fp = ((df['rule_prediction'] == 1) & (df['label'] == 0)).sum()
tn = ((df['rule_prediction'] == 0) & (df['label'] == 0)).sum()
fn = ((df['rule_prediction'] == 0) & (df['label'] == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print('--- Règle de détection beacon ---')
print(df[['src_ip', 'dst_ip', 'connection_count', 'recurrent_ratio', 'delta_cv', 'label', 'rule_prediction']].head(10).to_string(index=False))
print('\nMétriques :')
print(f'Vrais positifs  : {tp}')
print(f'Faux positifs  : {fp}')
print(f'Vrais négatifs : {tn}')
print(f'Faux négatifs  : {fn}')
print(f'Précision      : {precision:.4f}')
print(f'Rappel         : {recall:.4f}')
