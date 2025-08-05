import pandas as pd

df = pd.read_excel("SDG_11-1-1.xlsx")

# Step 1: Get most recent data per country
df_sorted = df.sort_values(['Country or Territory Name.1', 'Data Reference Year'], ascending=[True, False])
df_most_recent = df_sorted.drop_duplicates(subset='Country or Territory Name.1', keep='first')

# Step 2: Calculate the average of the lowest 50 slum percentages
lowest50_avg = (
    df_most_recent.sort_values('Proportion of urban population living in slums or informal settlements (%) (a)', ascending=True)
    .head(50)['Proportion of urban population living in slums or informal settlements (%) (a)']
    .mean()
)

# Step 3: Loop through every row and calculate score
scores = []

for index, row in df_most_recent.iterrows():
    year = row['Data Reference Year']
    slum_percent = row['Proportion of urban population living in slums or informal settlements (%) (a)']

    # Start score at 100
    score = 100

    # Penalize old data
    score -= max(0, 2024 - year)

    # Penalize based on difference from top 50 average
    penalty = slum_percent - lowest50_avg
    if penalty < 0:
        penalty = 0
    score -= penalty

    if not pd.isna(score):
        scores.append(round(score))
    else:
        scores.append(score)

# Step 4: Add scores as a new column
df_most_recent['Score'] = scores
df_most_recent = df_most_recent.sort_values('Score', ascending=False)
# Display top 10 results
print(df_most_recent[['Country or Territory Name.1', 'Score']].head(10))

variable_name = input("Enter Your Country:\n").strip()

# Get most recent row for this country
country_row = df_most_recent[df_most_recent['Country or Territory Name.1'] == variable_name]

if not country_row.empty:
    score = country_row['Score'].values[0]
    print("-------------------------------------")
    print(f"{variable_name} Score: {score:.2f}")
    print("-------------------------------------")
else:
    print("Country not found.")
