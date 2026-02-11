import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# from hmmlearn.hmm import GaussianHMM  # Uncomment for HMM option

regime_labels = {
    0: {'label': 'Bull', 'color': 'green'},
    1: {'label': 'Bear', 'color': 'red'},
    2: {'label': 'Sideways', 'color': 'blue'}
}

def build_features(df):
    features = pd.DataFrame(index=df.index)
    # Returns
    features['returns'] = df['Close'].pct_change()
    # Rolling volatility
    features['volatility'] = features['returns'].rolling(window=21).std()
    # Moving averages
    features['ma50'] = df['Close'].rolling(window=50).mean()
    features['ma200'] = df['Close'].rolling(window=200).mean()
    # Momentum indicator (e.g., RSI)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    features['rsi'] = 100 - (100 / (1 + rs))
    features = features.dropna()
    return features


def classify_regimes(features, method='kmeans'):
    # Use KMeans clustering by default
    X = features[['returns', 'volatility', 'rsi']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Map clusters to regimes based on mean returns/volatility
    cluster_stats = pd.DataFrame({
        'returns': features['returns'].groupby(clusters).mean(),
        'volatility': features['volatility'].groupby(clusters).mean()
    })
    # Assign Bull, Bear, Sideways
    bull = cluster_stats['returns'].idxmax()
    bear = cluster_stats['returns'].idxmin()
    sideways = [i for i in cluster_stats.index if i not in [bull, bear]][0]
    mapping = {bull: 0, bear: 1, sideways: 2}
    regimes = np.array([mapping[c] for c in clusters])
    return regimes

    # HMM option (commented)
    # if method == 'hmm':
    #     model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000)
    #     model.fit(X)
    #     hidden_states = model.predict(X)
    #     return hidden_states
