import pandas as pd
import json

# Read the mini data
df = pd.read_csv('local_test_data/mini_data.csv')

# Add unique_id (required by NeuralForecast)
df['unique_id'] = '1'

# We need input_size (160) + horizon (1) = 161 rows
# Let's take 165 just to be safe and show it works with more history
df_subset = df.head(165)

# Convert to list of dicts
instances = df_subset.to_dict(orient='records')

# Create the payload wrapper
payload = {"instances": instances}

# Save to json
output_path = 'request.json'
with open(output_path, 'w') as f:
    json.dump(payload, f, indent=2)

print(f"Created {output_path} with {len(instances)} instances")
