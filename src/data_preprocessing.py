import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    selected_features = [
        'concave points_worst', 'radius_worst', 'texture_worst',
        'concave points_mean', 'compactness_mean', 'area_worst',
        'perimeter_worst', 'concavity_mean', 'symmetry_worst',
        'fractal_dimension_mean', 'compactness_se', 'radius_se',
        'texture_se', 'area_se', 'smoothness_worst', 'symmetry_mean'
    ]

    X = df[selected_features]
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test