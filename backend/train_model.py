import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Create synthetic dataset
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    print('running training model.')
    data = {
        'ph': np.random.uniform(5.5, 7.5, n_samples),
        'nitrogen': np.random.uniform(20, 80, n_samples),
        'phosphorus': np.random.uniform(10, 60, n_samples),
        'potassium': np.random.uniform(15, 70, n_samples)
    }
    
    # Simple rules for crop assignment
    crops = []
    for i in range(n_samples):
        ph = data['ph'][i]
        n = data['nitrogen'][i]
        p = data['phosphorus'][i]
        k = data['potassium'][i]
        
        if ph < 6.0 and n > 60:
            crops.append(np.random.choice([0, 4]))  # Rice or Cotton
        elif 6.0 <= ph <= 7.0 and p > 40:
            crops.append(np.random.choice([1, 2]))  # Wheat or Maize
        elif ph > 7.0 and k > 50:
            crops.append(np.random.choice([3, 5]))  # Soybeans or Sugarcane
        else:
            crops.append(np.random.choice([6, 7, 8, 9]))  # Vegetables
    
    data['crop'] = crops
    return pd.DataFrame(data)

# Generate and prepare data
data = create_synthetic_data()
X = data[['ph', 'nitrogen', 'phosphorus', 'potassium']]
y = data['crop']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('soil_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'soil_model.pkl'")