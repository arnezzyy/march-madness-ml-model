from pathlib import Path
import pandas as pd

BASE_DIR = Path("/Users/arnnee/PycharmProjects/ProjectMadness")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

teams = pd.read_csv(DATA_DIR / "MTeams.csv")
submission = pd.read_csv(OUTPUT_DIR / "submission_stage2_v2.csv")

submission[["Season", "Team1", "Team2"]] = submission["ID"].str.split("_", expand=True)
submission["Season"] = submission["Season"].astype(int)
submission["Team1"] = submission["Team1"].astype(int)
submission["Team2"] = submission["Team2"].astype(int)

team1_names = teams.rename(columns={"TeamID": "Team1", "TeamName": "Team1_Name"})
team2_names = teams.rename(columns={"TeamID": "Team2", "TeamName": "Team2_Name"})

readable = submission.merge(team1_names[["Team1", "Team1_Name"]], on="Team1", how="left")
readable = readable.merge(team2_names[["Team2", "Team2_Name"]], on="Team2", how="left")

readable["Team1_WinPct"] = (readable["Pred"] * 100).round(1)
readable["Team2_WinPct"] = ((1 - readable["Pred"]) * 100).round(1)

readable["Predicted_Winner"] = readable.apply(
    lambda row: row["Team1_Name"] if row["Pred"] >= 0.5 else row["Team2_Name"],
    axis=1
)

readable = readable[
    ["Season", "Team1_Name", "Team2_Name", "Team1_WinPct", "Team2_WinPct", "Predicted_Winner"]
]

readable.to_csv(OUTPUT_DIR / "readable_predictions.csv", index=False)
print("Saved readable_predictions.csv")