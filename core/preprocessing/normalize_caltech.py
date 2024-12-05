from core.util.io import write_csv, read_csv, get_data_root

root = get_data_root()
output_filename = root / "processed/caltech_ev_sessions.csv"

data = read_csv("processed/caltech_ev_sessions.csv")

data["Consumption"] = (data["Consumption"] - data["Consumption"].min()) / (
    data["Consumption"].max() - data["Consumption"].min()
)

write_csv(data, "caltech_ev_sessions.csv")
