import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulated time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=200)
data = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.2, 200)  # Normal signal
data[50:60] += 3  # Introduce anomaly
df = pd.DataFrame({'date': dates, 'value': data})

# Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['value']])

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['value'], label='Data')
plt.scatter(df['date'][df['anomaly'] == -1], df['value'][df['anomaly'] == -1], 
            color='red', label='Anomalies', zorder=5)
plt.legend()
plt.title('Anomaly Detection in Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid()
plt.show()
