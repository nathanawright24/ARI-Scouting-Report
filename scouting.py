# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:13:17 2025

@author: natha
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm

seasons = [2019, 2020, 2021, 2022, 2023, 2024]
data = nfl.import_pbp_data(seasons)
arizonaplays = data[(data['posteam']=='ARI') | (data['defteam'] == 'ARI')]
offense = arizonaplays[arizonaplays['posteam']=='ARI']
defense = arizonaplays[arizonaplays['defteam']=='ARI']
drewpetzing_offense = offense[offense['season'].isin([2023, 2024])].copy()
kyler_plays = offense[offense['passer_player_id']=='00-0035228']

#-------------------------------------------------
# Offense Tendencies
def categorize_distance(ydstogo):
    if ydstogo <= 3:
        return 'short (1–3)'
    elif ydstogo <= 6:
        return 'medium (4–6)'
    elif ydstogo <= 10:
        return 'long (7–10)'
    else:
        return 'extra long (11+)'

def categorize_play(row):
    if row['play_type'] == 'pass':
        if row['pass_length'] == 'short':
            return 'Pass - Short'
        elif row['pass_length'] == 'deep':
            return 'Pass - Deep'
    elif row['play_type'] == 'run':
        gap = str(row['run_gap']).lower()
        location = str(row['run_location']).lower()
        if gap == 'guard' or location == 'middle':
            return 'Run - Inside'
        elif gap in ['tackle', 'end']:
            return 'Run - Outside'
    return None

offense_plays = drewpetzing_offense.copy()
offense_plays['distance_cat'] = offense_plays['ydstogo'].apply(categorize_distance)
offense_plays['play_cat'] = offense_plays.apply(categorize_play, axis=1)

filtered = offense_plays[
    offense_plays['play_cat'].notna() &
    offense_plays['distance_cat'].notna() &
    offense_plays['down'].notna()
]

grouped = filtered.groupby(['down', 'distance_cat', 'play_cat']).size().reset_index(name='play_count')
grouped['Percent'] = grouped.groupby(['down', 'distance_cat'])['play_count'].transform(lambda x: x / x.sum() * 100)

grouped['distance_cat'] = pd.Categorical(
    grouped['distance_cat'],
    categories=['short (1–3)', 'medium (4–6)', 'long (7–10)', 'extra long (11+)'],
    ordered=True
)

for down in sorted(grouped['down'].dropna().unique()):
    down_data = grouped[grouped['down'] == down]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=down_data,
        x='distance_cat',
        y='Percent',
        hue='play_cat',
        palette='Set2'
    )
    plt.title(f'Down {down}: Play Type % by Distance')
    plt.xlabel('Distance Bucket')
    plt.ylabel('Play Type %')
    plt.legend(title='Play Type')
    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------
# Formation
import pandas as pd

valid_offense = drewpetzing_offense[drewpetzing_offense['play_type'] != 'no_play'].copy()

def categorize_distance(ydstogo):
    if ydstogo <= 3:
        return 'short (1–3)'
    elif ydstogo <= 6:
        return 'medium (4–6)'
    elif ydstogo <= 10:
        return 'long (7–10)'
    else:
        return 'extra long (11+)'

valid_offense['distance_cat'] = valid_offense['ydstogo'].apply(categorize_distance)

# 1. Usage % formation
formation_usage = valid_offense['offense_formation'].value_counts(normalize=True).mul(100).reset_index()
formation_usage.columns = ['offense_formation', 'usage_percent']
plt.figure(figsize=(8, 5))
sns.barplot(
    data=formation_usage,
    x='offense_formation',
    y='usage_percent',
    palette='muted'
)
plt.title("Formation Usage Percentage")
plt.xlabel("Offense Formation")
plt.ylabel("Usage %")
plt.tight_layout()
plt.show()
# 2. Pass % formation
pass_by_formation = (
    valid_offense
    .groupby('offense_formation')['play_type']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .reset_index()
)
pass_by_formation['pass_percent'] = pass_by_formation['pass'] * 100
plt.figure(figsize=(8, 5))
sns.barplot(
    data=pass_by_formation,
    x='offense_formation',
    y='pass_percent',
    palette='Set2'
)
plt.title("Pass Percentage by Formation")
plt.xlabel("Offense Formation")
plt.ylabel("Pass %")
plt.tight_layout()
plt.show()

#Personnel each formation
def extract_personnel_counts(personnel):
    positions = {'RB': 0, 'TE': 0, 'WR': 0}
    if pd.isna(personnel):
        return pd.Series(positions)
    matches = re.findall(r'(\d+)\s*(RB|TE|WR)', personnel)
    for count, position in matches:
        positions[position] = int(count)
    return pd.Series(positions)

valid_offense[['RB_count', 'TE_count', 'WR_count']] = (
    valid_offense['offense_personnel']
    .apply(extract_personnel_counts)
)

# personnel_pkg
valid_offense['personnel_pkg'] = (
    valid_offense['RB_count'].astype(str) + 'RB_' +
    valid_offense['TE_count'].astype(str) + 'TE'
)

# Personnel usage within each formation
personnel_by_formation = (
    valid_offense
    .groupby(['offense_formation', 'personnel_pkg'])
    .size()
    .reset_index(name='count')
)
personnel_by_formation['percent'] = (
    personnel_by_formation
    .groupby('offense_formation')['count']
    .transform(lambda x: x / x.sum() * 100)
)

pivot = personnel_by_formation.pivot(
    index='offense_formation',
    columns='personnel_pkg',
    values='percent'
).fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
pivot.plot(kind='bar', ax=ax)
ax.set_title("Personnel Usage by Formation (RBs & TEs)")
ax.set_xlabel("Offense Formation")
ax.set_ylabel("Usage %")
ax.legend(title="Personnel Package", loc='upper right')

n_groups = pivot.shape[0]
for i in range(1, n_groups):
    ax.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
#--------------------------------------------------------
# Neutral Script Decisions

# no huddle
neutral_offense = valid_offense[
    (valid_offense['half_seconds_remaining'] > 240) &
    (valid_offense['score_differential'].abs() <= 16)
].copy()

no_huddle_counts = neutral_offense['no_huddle'].value_counts().rename(index={True: 'No Huddle', False: 'Huddle'})

plt.figure(figsize=(6, 6))
plt.pie(no_huddle_counts, labels=no_huddle_counts.index, autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
plt.title("No-Huddle vs Huddle Usage (Neutral Game Script)")
plt.tight_layout()
plt.show()

# 4th Down
# Distance buckets for 4th down (customizable)
# Define distance and field segment categories
def fourth_down_distance_cat(x):
    if x <= 2:
        return 'short (1–2)'
    elif x <= 5:
        return 'medium (3–5)'
    elif x <= 9:
        return 'long (6–9)'
    else:
        return 'extra long (10+)'

def fourth_down_field_position(yardline_100):
    if yardline_100 >= 80:
        return 'own territory'
    elif yardline_100 >= 60:
        return 'own gold zone'
    elif yardline_100 >= 40:
        return 'midfield'
    elif yardline_100 >= 20:
        return 'opponent gold zone'
    else:
        return 'red zone'

def classify_decision(row):
    if row['play_type'] in ['pass', 'run']:
        return 'go for it'
    elif row['play_type'] == 'field_goal':
        return 'field goal'
    elif row['play_type'] == 'punt':
        return 'punt'
    else:
        return 'other'

# Filter for 4th down in neutral game script
fourth_down = valid_offense[
    (valid_offense['down'] == 4) &
    (valid_offense['game_seconds_remaining'] > 240) &
    (valid_offense['score_differential'].abs() <= 16) &
    (valid_offense['play_type'] != 'no_play')
].copy()

# Categorize and classify decisions
fourth_down['distance_cat'] = fourth_down['ydstogo'].apply(fourth_down_distance_cat)
fourth_down['field_segment'] = fourth_down['yardline_100'].apply(fourth_down_field_position)
fourth_down['decision'] = fourth_down.apply(classify_decision, axis=1)
# Group and calculate percentage
decision_summary = (
    fourth_down.groupby(['distance_cat', 'field_segment', 'decision'])
    .size()
    .reset_index(name='count')
)
decision_summary['percent'] = decision_summary.groupby(
    ['distance_cat', 'field_segment']
)['count'].transform(lambda x: x / x.sum() * 100)

segment_order = [
    'own territory',   # furthest
    'own gold zone',
    'midfield',
    'opponent gold zone',
    'red zone'         # closest
]
decision_summary['field_segment'] = pd.Categorical(
    decision_summary['field_segment'],
    categories=segment_order,
    ordered=True
)

# 2. Create the FacetGrid
sns.set(style="whitegrid")
g = sns.catplot(
    data=decision_summary,
    x='field_segment',
    y='percent',
    hue='decision',
    col='distance_cat',
    kind='bar',
    palette='Set2',
    height=5,
    aspect=1.2,
    sharey=False      # allow each facet to scale independently, if you prefer
)

# 3. Tidy up titles and axis labels
g.set_titles("Distance: {col_name}")
g.set_axis_labels("Field Segment", "Decision %")
g.set_xticklabels(rotation=30)

# 4. Move legend above the plots
#    - Position: centered above the top row
#    - Adjust bbox to clear the bars
g._legend.set_title("4th Down Decision")
g._legend.set_loc('upper center')
g._legend.set_bbox_to_anchor((0.49, 1.055))  # x=0.5 center, y=1.05 slightly above

plt.tight_layout()
plt.show()

#---------------------------------------------------
# ---------------------Kyler------------------------
#---------------------------------------------------
kyler_plays = offense[offense['passer_player_id'] == '00-0035228']
kylerroutes = kyler_plays[
    (kyler_plays['play_type'] == 'pass') &
    (kyler_plays['route'].notna()) &
    (kyler_plays['cpoe'].notna())
].copy()

kyler_route_stats = kylerroutes.groupby('route').agg(
    avg_cpoe=('cpoe', 'mean')
).reset_index()
kyler_route_stats.to_csv("C:/Users/natha/Documents/BGA/Graduate Projects/ARIScoutingReport/kylerroutecpoe")

#------------------------------------------------
# Coverages
coverage_cpoe = kylerroutes.groupby('defense_coverage_type').agg(
    avg_cpoe=('cpoe', 'mean')
).reset_index()

plt.figure()
plt.bar(coverage_cpoe['defense_coverage_type'], coverage_cpoe['avg_cpoe'])
plt.title("Kyler Murray: CPOE by Coverage Type")
plt.xlabel("Coverage")
plt.ylabel("Avg CPOE")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

kyler_coverage_cpoe = kylerroutes[
    kylerroutes['cpoe'].notna() &
    kylerroutes['defense_man_zone_type'].notna()
]

man_zone_cpoe = kyler_coverage_cpoe.groupby('defense_man_zone_type')['cpoe'].mean().reset_index()
man_zone_cpoe.columns = ['Man/Zone', 'Avg CPOE']

plt.figure(figsize=(6, 4))
sns.barplot(data=man_zone_cpoe, x='Man/Zone', y='Avg CPOE', palette='coolwarm')
plt.title("Kyler Murray: CPOE vs Man & Zone Coverage")
plt.ylabel("Avg CPOE")
plt.xlabel("Coverage Type")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

#---------------------------------------------------
# 3x3 location grid

kylervalid = kylerroutes[
    kylerroutes['pass_location'].notna() &
    kylerroutes['ngs_air_yards'].notna()
].copy()

def bucket_air(air):
    if air < 9:
        return 'short (0–8.99)'
    elif air < 16:
        return 'medium (9–15.99)'
    else:
        return 'deep (16+)'

kylervalid['length_cat'] = kylervalid['ngs_air_yards'].apply(bucket_air)
kylervalid['location_cat'] = kylervalid['pass_location']  

heatmap_data = kylervalid.groupby(['length_cat', 'location_cat'])['cpoe'].mean().unstack()
heatmap_data = heatmap_data.reindex(
    index=[ 'deep (16+)', 'medium (9–15.99)','short (0–8.99)'],
    columns=['left', 'middle', 'right']
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="vlag",   
    center=0,
    cbar_kws={'label': 'Avg CPOE'}
)
plt.title("Kyler Murray: CPOE by Pass Length & Field Segment")
plt.xlabel("Pass Location")
plt.ylabel("Pass Distance Bucket")
plt.tight_layout()
plt.show()

#---------------------------------------------------
#-------------------- Chad Ryland ------------------
#---------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ryland_fgs = (
    data[
        data['season'].isin([2023, 2024]) &
        data['posteam'].isin(['NE', 'ARI']) &
        (data['play_type'] == 'field_goal') &
        (data['kicker_player_name'] == 'C.Ryland')
    ]
).copy()

ryland_fgs_summary = ryland_fgs.copy()

ryland_fgs_summary['made_binary'] = ryland_fgs_summary['field_goal_result'].apply(
    lambda x: 1 if x == 'made' else 0
)

ryland_fgs_summary['miss_prob_sum'] = (
    ryland_fgs_summary['no_score_prob'] +
    ryland_fgs_summary['opp_td_prob'] +
    ryland_fgs_summary['opp_fg_prob'] +
    ryland_fgs_summary['td_prob']
)

ryland_fgs_summary['expected_make_prob'] = ryland_fgs_summary['fg_prob']

kick_dist_range = np.arange(
    ryland_fgs_summary['kick_distance'].min(),
    ryland_fgs_summary['kick_distance'].max() + 1
)

X_actual = sm.add_constant(ryland_fgs_summary['kick_distance'])
y_actual = ryland_fgs_summary['made_binary']
logit_model_actual = sm.Logit(y_actual, X_actual)
result_actual = logit_model_actual.fit(disp=0)

X_pred = sm.add_constant(kick_dist_range)
smoothed_actual_probs = result_actual.predict(X_pred)

expected_agg = ryland_fgs_summary.groupby('kick_distance').agg(
    expected_make_prob_mean=('expected_make_prob', 'mean'),
    count=('expected_make_prob', 'size')
).reset_index()

X_expected = sm.add_constant(expected_agg['kick_distance'])
y_expected = expected_agg['expected_make_prob_mean']
weights = expected_agg['count']

glm_binom = sm.GLM(y_expected, X_expected, family=sm.families.Binomial(), freq_weights=weights)
result_expected = glm_binom.fit()

smoothed_expected_probs = result_expected.predict(X_pred)

plt.figure(figsize=(12, 7))

plt.plot(kick_dist_range, smoothed_actual_probs, label='Ryland Make Probability', color='blue', linewidth=2)
plt.plot(kick_dist_range, smoothed_expected_probs, label='Expected Make Probability', color='black', linewidth=2)

colors = ryland_fgs_summary['made_binary'].map({1: 'green', 0: 'red'})
plt.scatter(
    ryland_fgs_summary['kick_distance'],
    ryland_fgs_summary['made_binary'],
    c=colors,
    alpha=0.6,
    edgecolors='k',
    s=80,
    label='Actual Kicks (Made=Green, Missed=Red)'
)

plt.xlabel('Kick Distance (yards)')
plt.ylabel('Make Probability')
plt.title('Chad Ryland Field Goal Make Probability by Kick Distance (Smoothed)')
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.show()

#---------------------------------------------------
#-------------------- Defense ----------------------
#---------------------------------------------------
defensive_tendencies = defense[defense['season'].isin([2023, 2024])].copy()

# Coverage by Down & Distance
def bucket_ydstogo(ydstogo):
    if 1 <= ydstogo <= 4:
        return '1-4'
    elif 5 <= ydstogo <= 7:
        return '5-7'
    elif 8 <= ydstogo <= 10:
        return '8-10'
    else:
        return '11+'

defensive_tendencies['ydstogo_bucket'] = defensive_tendencies['ydstogo'].apply(bucket_ydstogo)

coverage_types = defensive_tendencies['defense_coverage_type'].unique()
coverage_order = ['COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 'COVER_6', 'PREVENT']
colors = ['#cdb4db', '#b5ead7', '#ffdac1', '#ffe5d9', '#e2f0cb', '#f9c6c9', '#f6e2b3']

ydstogo_buckets = ['1-4', '5-7', '8-10', '11+']
downs = [1, 2, 3, 4]

for down in downs:
    df_down = defensive_tendencies[defensive_tendencies['down'] == down]
    
    pivot_counts = pd.pivot_table(
        df_down,
        index='ydstogo_bucket',
        columns='defense_coverage_type',
        values='play_id',
        aggfunc='count',
        fill_value=0
    )
    
    pivot_percents = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
    pivot_percents = pivot_percents.reindex(ydstogo_buckets).fillna(0)
    pivot_percents = pivot_percents.reindex(columns=coverage_order).fillna(0)
    
    ax = pivot_percents.plot(kind='bar', figsize=(10,6), color=colors)
    plt.title(f'Defensive Coverage Distribution by Yards to Go Buckets - Down {down}')
    plt.xlabel('Yards to Go Bucket')
    plt.ylabel('Coverage Type Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(title='Coverage Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------
# Man Zone
pass_plays = defense[(defense['play_type'] == 'pass') & (defense['season'].isin([2023, 2024]))].copy()

man_zone_counts = pass_plays['defense_man_zone_type'].value_counts()

categories = ['MAN_COVERAGE', 'ZONE_COVERAGE']
counts = [man_zone_counts.get(cat, 0) for cat in categories]

colors = ['#97233F', '#000000']

def make_autopct(colors):
    def my_autopct(pct):
        return ('%.1f%%' % pct) if pct > 0 else ''
    return my_autopct

plt.figure(figsize=(7,7))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=categories,
    autopct=make_autopct(colors),
    startangle=140,
    colors=colors,
    textprops={'fontsize': 14}
)
autotexts[1].set_color('white')
plt.title('Man vs Zone Coverage Splits on Pass Plays (ARI Defense, 2023-24)')
plt.show()
#----------------------------------------------------------
# Pass Rush
pass_plays = defense[(defense['play_type'] == 'pass') & (defense['season'].isin([2023, 2024]))].copy()

rush_counts = pass_plays['number_of_pass_rushers'].value_counts().sort_index()
rush_percent = (rush_counts / rush_counts.sum()) * 100

plt.figure(figsize=(10,6))
plt.bar(rush_percent.index, rush_percent.values, color='#97233F')
plt.xlabel('Number of Pass Rushers')
plt.ylabel('Percentage of Pass Plays')
plt.title('Percentage Distribution of Number of Pass Rushers on Pass Plays (ARI Defense, 2023-24)')
plt.xticks(rush_percent.index)
plt.ylim(0, rush_percent.max() + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#------
blitz_count = len(pass_plays[pass_plays['number_of_pass_rushers'] > 4.0])
non_blitz_count = len(pass_plays) - blitz_count

labels = ['Blitz', 'Non-Blitz']
counts = [blitz_count, non_blitz_count]
colors = ['#97233F', '#000000']

plt.figure(figsize=(7,7))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 14}
)
autotexts[1].set_color('white')
plt.title('Blitz vs Non-Blitz on Pass Plays (ARI Defense, 2023-24)')
plt.show()
#----------------
pass_plays = defense[(defense['play_type'] == 'pass') & (defense['season'].isin([2023, 2024]))].copy()

blitz_mask = pass_plays['number_of_pass_rushers'] > 4.0

formation_counts = pass_plays['offense_formation'].value_counts()
blitz_counts = pass_plays[blitz_mask]['offense_formation'].value_counts()

blitz_percent_by_formation = (blitz_counts / formation_counts).fillna(0) * 100
blitz_percent_by_formation = blitz_percent_by_formation.sort_values(ascending=False)

plt.figure(figsize=(12,6))
plt.bar(blitz_percent_by_formation.index, blitz_percent_by_formation.values, color='#97233F')
plt.xlabel('Offense Formation')
plt.ylabel('Blitz Percentage (%)')
plt.title('Blitz Percentage on Pass Plays by Offense Formation (ARI Defense, 2023-24)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, blitz_percent_by_formation.max() + 10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#----------------------------------------------------------
# cpoe allowed
pass_plays = defense[(defense['play_type'] == 'pass') & (defense['season'].isin([2023, 2024]))].copy()

cpoe_by_route = pass_plays.groupby('route')['cpoe'].mean().sort_values()

plt.figure(figsize=(14,7))
bars = plt.bar(cpoe_by_route.index, cpoe_by_route.values, color='#97233F')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.xlabel('Route')
plt.ylabel('Average CPOE Allowed')
plt.title('Average CPOE Allowed by Defense by Route (ARI Defense, 2023-24)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
