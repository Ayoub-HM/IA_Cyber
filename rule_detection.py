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

# ---------------------------------------------------
# 9.1 - Séparation train / test
# ---------------------------------------------------
from sklearn.model_selection import train_test_split

features = [
    'connection_count',
    'delta_mean_s',
    'delta_std_s',
    'delta_cv',
    'recurrent_ratio',
    'bytes_out_mean',
    'bytes_out_cv',
    'bytes_in_mean',
    'duration_mean_ms',
    'packets_out_mean',
    'tls_ratio',
    'night_ratio',
    'unique_ports'
]

X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('\n--- Séparation train/test ---')
print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape :', y_test.shape)
