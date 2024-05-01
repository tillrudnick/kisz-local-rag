import pandas as pd

import pandas as pd

def load_json(path):
    # Using 'sep=";"' to specify the delimiter, and 'quotechar='"' to handle quotes appropriately.
    # Additionally, specify 'engine="python"' if 'c' engine fails to parse the file correctly.
    df = pd.read_csv(path, sep=';', quotechar='"', engine='python')

    # Convert DataFrame to JSON
    json_data = df.to_json(orient='records')

    # Print the first JSON record
    # Since json_data is a JSON string, we need to parse it back to access elements if necessary
    import json
    records = json.loads(json_data)  # Parse the JSON string back into a list
    return records

def format_json_to_string(data):
    # List to hold formatted key-value pairs
    formatted_pairs = []

    # Iterate over each key-value pair in the dictionary
    for key, value in data.items():
        # Format the key-value pair as "key: value"
        # If the value is None, represent it as 'null'
        formatted_pair = f"{key}: {value if value is not None else 'null'}"
        formatted_pairs.append(formatted_pair)

    # Join all formatted pairs with a comma and a space
    result_string = ', '.join(formatted_pairs)
    return result_string

def load_csv():
    path = './sample_data/wb_utf8.csv'
    json_data = load_json(path)
    elements = [format_json_to_string(item) for item in json_data]
    return elements
