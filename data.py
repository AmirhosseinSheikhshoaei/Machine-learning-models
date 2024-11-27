import pandas as pd

# Load the data
data = pd.read_csv(r"D:\AR\n.csv")

# Separate input & outputs
X = data.iloc[:, :5].values  # Assuming the first 4 columns are features
y = data.iloc[:, 5:7].values  # Select columns 5 to end for the target (multiple columns)

