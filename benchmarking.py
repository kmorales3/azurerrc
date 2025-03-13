import time
import pandas as pd
import json
import requests
import os

# Define test cases (same set for every benchmark)
TEST_CARS = [
    ("BNSF", "238515"),
    ("BNSF", "238571"),
    ("BNSF", "238804"),
    ("BNSF", "238805"),
    ("BNSF", "238864"),
    ("BNSF", "270598"),
    ("BNSF", "270647"),
    ("BNSF", "270846"),
    ("DTTX", "459385"),
    ("DTTX", "62579"),
    ("DTTX", "723873"),
    ("DTTX", "73030"),
    ("DTTX", "747095"),
    ("DTTX", "748863"),
    ("DTTX", "760434"),
    ("DTTX", "767861"),
    ("BNSF", "255125"),
    ("BNSF", "255236"),
    ("DTTX", "723755"),
    ("DTTX", "766537"),
    ("FEC", "72191"),
    ("UCRY", "57122"),
    ("BNSF", "238933"),
    ("BNSF", "239425"),
    ("DTTX", "61505"),
    ("DTTX", "741316"),
    ("DTTX", "743399"),
    ("DTTX", "747295"),
    ("DTTX", "750869"),
    ("SFLC", "9303"),
    ("AOK", "74653"),
    ("BNSF", "231178"),
    ("BNSF", "256062"),
    ("DTTX", "449826"),
    ("DTTX", "732274"),
    ("DTTX", "760029"),
    ("BNSF", "238379"),
    ("BNSF", "239612"),
    ("DTTX", "53092"),
    ("DTTX", "751411"),
    ("DTTX", "759422"),
]  # Adjust list based on real-world data

NUM_RUNS = 2  # Number of times to query each car
results = []


def execute_notebook(databricks_instance, token, notebook_path, car_id,
                        car_number):
    print(f'Executing notebook: {notebook_path} with car_id: {car_id}-{car_number}')
    
    # API endpoint for running a notebook
    url = f"{databricks_instance}/api/2.0/jobs/runs/submit"

    # Define the payload for the API request
    payload = {
        "run_name": "Car Query Run",
        "existing_cluster_id": "0422-200925-pints678",
        "notebook_task": {
            "notebook_path": notebook_path,
            "base_parameters": {"car_id": car_id, "car_number":
                                car_number},
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Run the notebook
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.text}")

    run_id = response.json().get("run_id")
    if not run_id:
        raise Exception("Failed to get run_id from the response")

    # Check job status
    while True:
        status_url = f"{databricks_instance}/api/2.0/jobs/runs/get?run_id={run_id}"
        status_response = requests.get(status_url, headers=headers)
        run_status = status_response.json()

        life_cycle_state = run_status["state"]["life_cycle_state"]

        if life_cycle_state in ["TERMINATED", "SKIPPED", "FAILED"]:
            # Job has completed
            break

        # Sleep for a while before checking again
        time.sleep(5)  # Wait for 5 seconds before the next check

    # Retrieve results from the output
    output_url = f"{databricks_instance}/api/2.0/jobs/runs/get-output?run_id={run_id}"
    output_response = requests.get(output_url, headers=headers)
    return output_response.json()


for car_id, car_number in TEST_CARS:
    for i in range(NUM_RUNS):
        start_time = time.time()

        ntb_results = execute_notebook(databricks_instance, databricks_token, db_notebook_path, car_id, car_number)
        # ntb_results = json.loads(ntb_results["notebook_output"]["result"].strip("'"))[0]
        try:
            train_data = ntb_results["notebook_output"]["result"].split("'")
            trn_id = train_data[1]
            trn_dst_cty = train_data[3]
            trn_dest_st = train_data[5]

        except KeyError:
            trn_id, trn_dst_cty, trn_dest_st = "SymbolNotFound", "NA", "NA"

 
        
        
        # try:
        #     trn_id = ntb_results["trn_id"]
        #     trn_dst_cty = ntb_results["dest_city_frefrm"]
        #     trn_dest_st = ntb_results["dest_st"]
        # except KeyError:
        #     # Handle missing data or bad JSON responses
        #     trn_id, trn_dst_cty, trn_dest_st = "SymbolNotFound", "NA", "NA"

        execution_time = time.time() - start_time
        
        # Append results to the list
        results.append({
            "car_id": car_id,
            "car_number": car_number,
            "trn_id": trn_id,
            "trn_dst_cty": trn_dst_cty,
            "trn_dest_st": trn_dest_st,
            "execution_time": execution_time
        })

        print(f"Run {i+1}/{NUM_RUNS} - Car: {car_id}-{car_number} - Symbol: {trn_id} - Destination: {trn_dst_cty}-{trn_dest_st} - Time: {execution_time:.2f}s")

# Convert results to a Pandas DataFrame
benchmark_df = pd.DataFrame(results)

# Save to CSV
csv_filename = "new_benchmark_results.csv"
benchmark_df.to_csv(csv_filename, index=False)

print(f"Benchmarking complete! Results saved to {csv_filename}")