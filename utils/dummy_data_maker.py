import numpy as np
import pandas as pd

np.random.seed(42)

ages = np.random.randint(20, 60, 500) # generate 500 data points for age from 20 - 60


salaries = 3000 + (ages * 500) # + np.random.normal(0, 5000, 500) # base salary + age_component(experience) + noise

# Create a DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Salary': salaries.astype(int)  # optional: round to int
})

# Save to CSV
df.to_csv('age_salary_data.csv', index=False)