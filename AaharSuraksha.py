!pip install pandas numpy scikit-learn matplotlib seaborn django
import pandas as pd
data = pd.read_csv("LIBS_spectra_dry_potatoes_withcolumns.csv")
print(data)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Prepare the Data
# Define frequency ranges and corresponding pesticide categories
data = {
    'Frequency': [
        (100, 178), (180, 190), (200, 210), (340, 360),
        (400, 410), (420, 450), (470, 490), (510, 555),
        (580, 601), (666, 690), (876, 999)
    ],
    'Category': [
        'Acephate (C)', 'Alachlor (B2)', 'Atrazine (C)', 'Benomyl (C)',
        'Bifenthrin (C)', 'Captan (B2)', 'Chlorothalonil (B2)',
        'Cypermethrin (C)', 'Dichlorvos (C)', 'Diclofop-Methyl (C)',
        'Dicofol (C)'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 2: Feature Engineering
# Create a feature for each frequency range
for index, row in df.iterrows():
    df.at[index, 'Range'] = f"{row['Frequency'][0]}-{row['Frequency'][1]}"

# Create a mapping of frequency to category
frequency_to_category = {}
for index, row in df.iterrows():
    frequency_to_category[row['Range']] = row['Category']

# Generate example frequency data for classification
# Here, we simulate some frequency data points for testing
frequencies = np.random.randint(100, 1000, size=1000)  # Random frequencies between 100 and 999
categories = []

# Assign categories based on frequency ranges
for freq in frequencies:
    for range_str, category in frequency_to_category.items():
        low, high = map(int, range_str.split('-'))
        if low <= freq <= high:
            categories.append(category)
            break
    else:
        categories.append('Unknown')  # If no category is found

# Create a DataFrame for the generated data
classification_data = pd.DataFrame({'Frequency': frequencies, 'Category': categories})

# Step 3: Split the Data
X = classification_data[['Frequency']]
y = classification_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('LIBS_spectra_dry_potatoes_withcolumns.csv')  # Replace with your actual file path

# Check the first few rows and the columns
print(data.head())
print(data.columns)

# Identify the correct target variable name
# For example, if the target variable is 'class', use that
target_variable = 'class'  # Change this to the actual name if different

# Preprocess the data
X = data.drop(columns=[target_variable])  # Features
y = data[target_variable]  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))