import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# I use dis to generate the pretty plots of the various csv data <3

# Load data
data = pd.read_csv("w_d_codes.csv")

# Convert "Observed" column to datetime
data["Observed"] = pd.to_datetime(data["Observed"])

# Resample the data to daily median based on "Observed" column
daily_median = data.resample("D", on="Observed").median()

# Set the figure size for better visibility
plt.figure(figsize=(10, 5))

# Plot the daily median of Humidity
plt.plot(
    daily_median.index,
    daily_median["Temperature"],
    label="Daily Median Temperature",
    color="blue",
    linewidth=1,
    alpha=0.8,
)
# plt.plot(data["Observed"], data["Rainfall"],
#        label="Daily Median Rainfall", color="blue", linewidth=1, alpha=0.8)
# plt.plot(daily_median.index, daily_median["Wind Speed"],
#        label="Daily Median Wind Speed", color="blue", linewidth=1, alpha=0.8)
# plt.plot(daily_median.index, daily_median["Humidity"],
#        label="Daily Median Humidity", color="purple", linewidth=1, alpha=0.8)
plt.ylabel("Humidity (°C)")

# Formatting the x-axis for better visibility
# Show ticks every 2 weeks (adjust as needed)
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=5))
# Show dates in Year-Month-Day format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Rotate the x-axis labels for better readability and set font size
plt.xticks(rotation=45, ha="right", fontsize=10)

# Add gridlines
plt.grid(True, linestyle="--", alpha=0.5)

# Add a legend with larger font size
plt.legend(fontsize=12)

# Improve layout
plt.tight_layout()

# Change between .jpg and .pdf for your needs
plt.savefig("./plots/Temperature.pdf")
# plt.savefig("./plots/Rainfall.pdf")
# plt.savefig("./plots/Wind_Speed.pdf")
# plt.savefig("./plots/Humidity.pdf")

# Show plot
# plt.show()
