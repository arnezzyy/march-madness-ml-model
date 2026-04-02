clc;
clear;
close all;

% FILE PATHS
baseDir = "/Users/arnnee/PycharmProjects/ProjectMadness";

teamsFile = fullfile(baseDir, "data", "MTeams.csv");
predFile = fullfile(baseDir, "output", "readable_predictions.csv");

% LOAD DATA
teams = readtable(teamsFile, 'TextType', 'string');
preds = readtable(predFile, 'TextType', 'string');

teamNames = sort(unique(teams.TeamName));

% DROPDOWN SELECTIONS
[idx1, tf1] = listdlg( ...
    'PromptString', 'Select Team 1:', ...
    'SelectionMode', 'single', ...
    'ListString', cellstr(teamNames), ...
    'ListSize', [300 400]);

if ~tf1
    error('No Team 1 selected.');
end

team1 = teamNames(idx1);

remainingTeams = teamNames(teamNames ~= team1);

[idx2, tf2] = listdlg( ...
    'PromptString', 'Select Team 2:', ...
    'SelectionMode', 'single', ...
    'ListString', cellstr(remainingTeams), ...
    'ListSize', [300 400]);

if ~tf2
    error('No Team 2 selected.');
end

team2 = remainingTeams(idx2);

% FIND MATCHUP
match1 = preds.Team1_Name == team1 & preds.Team2_Name == team2;
match2 = preds.Team1_Name == team2 & preds.Team2_Name == team1;

if any(match1)
    row = preds(match1, :);
    prob1 = row.Team1_WinPct(1);
    prob2 = row.Team2_WinPct(1);
    name1 = row.Team1_Name(1);
    name2 = row.Team2_Name(1);
elseif any(match2)
    row = preds(match2, :);
    prob1 = row.Team2_WinPct(1);
    prob2 = row.Team1_WinPct(1);
    name1 = team1;
    name2 = team2;
else
    error("Matchup not found in readable_predictions.csv");
end

% DISPLAY TEXT OUTPUT
fprintf('\nPredicted Matchup:\n');
fprintf('%s vs %s\n', name1, name2);
fprintf('%s win probability: %.1f%%\n', name1, prob1);
fprintf('%s win probability: %.1f%%\n', name2, prob2);

if prob1 > prob2
    winner = name1;
elseif prob2 > prob1
    winner = name2;
else
    winner = "Toss-up";
end

fprintf('Predicted winner: %s\n', winner);

% BAR CHART
figure('Name', 'March Madness Matchup Prediction', 'NumberTitle', 'off');
bar([prob1, prob2]);
set(gca, 'XTickLabel', {char(name1), char(name2)});
ylabel('Win Probability (%)');
title(sprintf('%s vs %s', name1, name2));
ylim([0 100]);
grid on;

text(1, prob1 + 2, sprintf('%.1f%%', prob1), ...
    'HorizontalAlignment', 'center', 'FontSize', 11);
text(2, prob2 + 2, sprintf('%.1f%%', prob2), ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

annotation('textbox', [0.35 0.8 0.3 0.08], ...
    'String', sprintf('Predicted Winner: %s', winner), ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');