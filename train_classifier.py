import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data (ensure data.pickle is created correctly during feature extraction)
data_dict = pickle.load(open('./datawords.pickle', 'rb'))

data = np.asarray(data_dict['datawords'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model
f = open('modelwords.p', 'wb')
pickle.dump({'modelwords': model}, f)
f.close()
