print(rag_query("Unemployment rate in United States in 2019"))

print(rag_query("Unemployment rate in United States in 2019"))

import pandas as pd
import io

# Assuming 'documents' list contains the CSV string
# We take the first document, as it contains our dataset
csv_data = documents[0]

# Read the CSV string into a pandas DataFrame
df_context = pd.read_csv(io.StringIO(csv_data))

# Filter for United States in 2019
# Ensure column names match your CSV (e.g., 'Country', 'Year', 'Unemployment rates')
filtered_df = df_context[(df_context['Country'] == 'United States') & (df_context['Year'] == 2019)]

# Extract the 'Unemployment rates' value
if not filtered_df.empty:
    unemployment_rate = filtered_df['Unemployment rates'].iloc[0]
    print(f"Programmatically extracted unemployment rate for United States in 2019: {unemployment_rate}%")
else:
    print("Data not found for United States in 2019.")

# You can then incorporate this extracted value directly into your LLM prompt, or present it as the answer, rather than relying on the LLM to extract it from the raw text context. This approach is particularly useful for precise data points that are critical for your application.

#Task
T#he extracted unemployment rate for the United States in 2019 is 3.7%.

#Extract Accurate Numerical Value
#Subtask:
#Programmatically extract the exact unemployment rate for the United States in 2019 from the loaded CSV data using pandas. This step is crucial for ensuring the accuracy of the numerical information.

#Generate Qwen Prompt with Extracted Value
#Subtask:
#Construct a new prompt for Qwen that explicitly includes the accurate unemployment rate obtained in the previous step. The prompt will ask Qwen to formulate a natural language answer incorporating this specific value from the context.

#Reasoning: Construct a new prompt for Qwen that explicitly includes the extracted unemployment rate and instructs Qwen to use this value in its answer.



new_query = "Unemployment rate in United States in 2019"
# Build prompt using the extracted value
new_prompt = f"""You are a helpful assistant.

Here is a factual statement: The unemployment rate in the United States in 2019 is {unemployment_rate}% based on the provided data.

Question: {new_query}

# Reasoning: The previous cell failed because 'unemployment_rate' was not defined. This variable is defined in cell 'd009007b'. To fix this, I will combine the code from cell 'd009007b' and cell 'bdce828a' into a single code block to ensure 'unemployment_rate' is properly defined before being used.
Based on the factual statement provided, what was the unemployment rate in the United States in 2019? Provide a concise answer using the given numerical value.

Answer:
"""

print("Generated new Qwen prompt with explicit value:")
print(new_prompt)

import pandas as pd
import io

# Assuming 'documents' list contains the CSV string
# We take the first document, as it contains our dataset
csv_data = documents[0]

# Read the CSV string into a pandas DataFrame
df_context = pd.read_csv(io.StringIO(csv_data))

# Filter for United States in 2019
# Ensure column names match your CSV (e.g., 'Country', 'Year', 'Unemployment rates')
filtered_df = df_context[(df_context['Country'] == 'United States') & (df_context['Year'] == 2019)]

# Extract the 'Unemployment rates' value
if not filtered_df.empty:
    unemployment_rate = filtered_df['Unemployment rates'].iloc[0]
    print(f"Programmatically extracted unemployment rate for United States in 2019: {unemployment_rate}%")
else:
    unemployment_rate = "N/A" # Define a fallback value
    print("Data not found for United States in 2019.")

new_query = "Unemployment rate in United States in 2019"
# Build prompt using the extracted value
new_prompt = f"""You are a helpful assistant.

Here is a factual statement: The unemployment rate in the United States in 2019 is {unemployment_rate}% based on the provided data.

Question: {new_query}

Based on the factual statement provided, what was the unemployment rate in the United States in 2019? Provide a concise answer using the given numerical value.

Answer:
"""

print("Generated new Qwen prompt with explicit value:")
print(new_prompt)

# Query Qwen for Final Answer
#Subtask:
#Execute Qwen with the specially crafted prompt to generate the final answer, ensuring that Qwen uses the correct numerical value provided to it.

#Reasoning: The user wants to execute the rag_query function with the new_prompt and print the result, ensuring show_sources is set to False.
