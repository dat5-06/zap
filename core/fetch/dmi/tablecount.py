from matplotlib import pyplot as plt
from core.util.io import read_csv, save_fig

# TODO: This file currently does not get access to the weather code data.
# This file is currently deprecated.

# I use dis to count the weather codes and show in not so cool table

# Load data
file_path = "core/data/external/w_d_codes.csv"
data = read_csv(file_path)


# Count the occurrences of each weather code
weather_code_counts = data["Weather Codes"].value_counts().reset_index()

# Rename the columns for clarity
weather_code_counts.columns = ["Weather Code", "Count"]

# Sort the table by the count in descending order
weather_code_counts = weather_code_counts.sort_values(by="Count", ascending=False)

# Create a figure and axis to display the table
# Adjust the size of the figure for better readability
fig, ax = plt.subplots(figsize=(8, 4))

# Hide the axis
ax.axis("tight")
ax.axis("off")

# Create the table and add it to the figure
table = ax.table(
    cellText=weather_code_counts.values,
    colLabels=weather_code_counts.columns,
    cellLoc="center",
    loc="center",
)

# Format the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust the scale of the table for better readability

# Save the figure as a PDF file
output_path = "core/fetch/dmi/dmi_plot/Weather_Code_Table.pdf"
save_fig(output_path)

# Optionally, show the plot
# plt.show()
