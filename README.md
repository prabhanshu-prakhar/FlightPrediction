Data Collection
The training and test datasets containing flight information are loaded into Pandas DataFrames from Excel files.
import pandas as pd

# Load training data
df_train = pd.read_excel('Data_Train.xlsx')

# Load test data
df_test = pd.read_excel('Test_set.xlsx')

Data Preprocessing
The data is preprocessed to handle missing values and convert data types. Initial data cleaning steps are performed.\
# Drop missing values
final_df.dropna(inplace=True)

# Convert data types
final_df['Date'] = final_df['Date'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)


Feature Engineering
New features are created from existing data to improve the predictive performance of the model.
# Extract features from arrival time
final_df['Arrival_hour'] = final_df['Arrival_Time'].str.split(':').str[0]
final_df['Arrival_min'] = final_df['Arrival_Time'].str.split(':').str[1]


Machine Learning
Various machine learning algorithms are trained and evaluated to predict flight prices.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split data into train and test sets
X = final_df.drop('Price', axis=1)
y = final_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)



