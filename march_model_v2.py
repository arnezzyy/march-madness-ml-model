from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/Users/arnnee/PycharmProjects/ProjectMadness")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# LOAD DATA
regular = pd.read_csv(DATA_DIR / "MRegularSeasonCompactResults.csv")
tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
sample = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")

# CLEAN SEEDS
seeds = seeds.copy()
seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
seeds = seeds[["Season", "TeamID", "SeedNum"]]


# BUILD TEAM STATS
wins = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
wins.columns = ["Season", "TeamID", "PF", "PA"]
wins["Win"] = 1

losses = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
losses.columns = ["Season", "TeamID", "PF", "PA"]
losses["Win"] = 0

games = pd.concat([wins, losses], ignore_index=True)
games["PointDiff"] = games["PF"] - games["PA"]

team_stats = (
    games.groupby(["Season", "TeamID"], as_index=False)
    .agg(
        WinPct=("Win", "mean"),
        PointDiff=("PointDiff", "mean")
    )
)


# BUILD TRAINING DATA
rows = []

for _, row in tourney.iterrows():
    t1 = int(min(row["WTeamID"], row["LTeamID"]))
    t2 = int(max(row["WTeamID"], row["LTeamID"]))
    target = 1 if int(row["WTeamID"]) == t1 else 0

    rows.append({
        "Season": int(row["Season"]),
        "Team1": t1,
        "Team2": t2,
        "Target": target
    })

train = pd.DataFrame(rows)

# MERGE FEATURES
team1_stats = team_stats.rename(columns={
    "TeamID": "Team1",
    "WinPct": "Team1_WinPct",
    "PointDiff": "Team1_PointDiff"
})

team2_stats = team_stats.rename(columns={
    "TeamID": "Team2",
    "WinPct": "Team2_WinPct",
    "PointDiff": "Team2_PointDiff"
})

seed1 = seeds.rename(columns={
    "TeamID": "Team1",
    "SeedNum": "Team1_Seed"
})

seed2 = seeds.rename(columns={
    "TeamID": "Team2",
    "SeedNum": "Team2_Seed"
})

train = train.merge(team1_stats, on=["Season", "Team1"], how="left")
train = train.merge(team2_stats, on=["Season", "Team2"], how="left")
train = train.merge(seed1, on=["Season", "Team1"], how="left")
train = train.merge(seed2, on=["Season", "Team2"], how="left")

train["Team1_Seed"] = train["Team1_Seed"].fillna(20)
train["Team2_Seed"] = train["Team2_Seed"].fillna(20)

# FEATURE DIFFERENCES
train["SeedDiff"] = train["Team1_Seed"] - train["Team2_Seed"]
train["WinPctDiff"] = train["Team1_WinPct"] - train["Team2_WinPct"]
train["PointDiffDiff"] = train["Team1_PointDiff"] - train["Team2_PointDiff"]

features = ["SeedDiff", "WinPctDiff", "PointDiffDiff"]

X = train[features].fillna(0)
y = train["Target"]


# TRAIN / VALIDATE MODEL
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

valid_preds = model.predict_proba(X_valid)[:, 1]
print("Validation log loss:", round(log_loss(y_valid, valid_preds), 5))

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nFeature Importance:")
print(coef_df)

print("\nModel trained.")


sub = sample.copy()
sub[["Season", "Team1", "Team2"]] = sub["ID"].str.split("_", expand=True)
sub["Season"] = sub["Season"].astype(int)
sub["Team1"] = sub["Team1"].astype(int)
sub["Team2"] = sub["Team2"].astype(int)

# Force Team1 < Team2 to match training convention
mask = sub["Team1"] > sub["Team2"]
sub.loc[mask, ["Team1", "Team2"]] = sub.loc[mask, ["Team2", "Team1"]].values

sub = sub.merge(team1_stats, on=["Season", "Team1"], how="left")
sub = sub.merge(team2_stats, on=["Season", "Team2"], how="left")
sub = sub.merge(seed1, on=["Season", "Team1"], how="left")
sub = sub.merge(seed2, on=["Season", "Team2"], how="left")

sub["Team1_Seed"] = sub["Team1_Seed"].fillna(20)
sub["Team2_Seed"] = sub["Team2_Seed"].fillna(20)

sub["SeedDiff"] = sub["Team1_Seed"] - sub["Team2_Seed"]
sub["WinPctDiff"] = sub["Team1_WinPct"] - sub["Team2_WinPct"]
sub["PointDiffDiff"] = sub["Team1_PointDiff"] - sub["Team2_PointDiff"]

X_sub = sub[features].fillna(0)
preds = model.predict_proba(X_sub)[:, 1]

submission = pd.DataFrame({
    "ID": sample["ID"],
    "Pred": preds.clip(0.025, 0.975)
})

out_file = OUTPUT_DIR / "submission_stage2_v2.csv"
submission.to_csv(out_file, index=False)

print(f"\nSubmission file created: {out_file}")