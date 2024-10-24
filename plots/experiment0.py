import matplotlib.pyplot as plt
import numpy as np
from core.util.io import read_csv

park_data = read_csv(
    "/home/max/Desktop/uni/5. semester/zap/core/data/processed/park_timeseries_lin.csv"
)

# Normalize the data
min_value = park_data["Ladepark 5"].min()
max_value = park_data["Ladepark 5"].max()
normalized_data = (park_data["Ladepark 5"] - min_value) / (max_value - min_value)

data_len = len(normalized_data)
train_end = int(0.8 * data_len)
test_end = train_end + int(0.1 * data_len)

data_hours = np.arange(data_len)

# Plot the training data
plt.plot(
    data_hours[:train_end],
    normalized_data[:train_end],
    color="blue",
    label="Training",
)

# Plot the validation data
plt.plot(
    data_hours[train_end:test_end],
    normalized_data[train_end:test_end],
    color="orange",
    label="Validation",
)

# Plot the test data
plt.plot(
    data_hours[test_end:],
    normalized_data[test_end:],
    color="green",
    label="Test",
)

ax = plt.gca()
ax.legend(loc="lower center", ncol=3)

plt.xlabel("Data Hours")
plt.ylabel("kWh normalized")

plt.savefig("park5")
