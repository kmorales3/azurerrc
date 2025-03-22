import time
import pandas as pd
import json
import os
import requests
import asyncio
import aiohttp


# async def execute_parallel(databricks_instance, token, notebook_path, test_cases):
#     """Runs all notebook executions in parallel and maps car IDs to run IDs."""
#     tasks = [
#         execute_notebook(databricks_instance, token, notebook_path, car_id, car_number)
#         for car_id, car_number in test_cases
#     ]
#     run_id_list = await asyncio.gather(*tasks)  # Run all at once

#     # Convert list of dictionaries into a dictionary mapping car_number -> run_id
#     run_id_map = {
#         f"{car_id}-{car_number}": run_id_dict["run_id"] for (car_id, car_number), run_id_dict in zip(test_cases, run_id_list)
#     }

#     return run_id_map  # Now it's a proper dictionary


# async def execute_notebook(databricks_instance, token, notebook_path, car_id, car_number):
#     """Asynchronously executes the Databricks notebook via API and returns a dictionary with run_id."""
#     print(f'Executing notebook: {notebook_path} with car_id: {car_id}-{car_number}')
    
#     url = f"{databricks_instance}/api/2.0/jobs/runs/submit"

#     payload = {
#         "run_name": "Car Query Run",
#         "existing_cluster_id": "0422-200925-pints678",
#         "notebook_task": {
#             "notebook_path": notebook_path,
#             "base_parameters": {"car_id": car_id, "car_number": car_number},
#         },
#     }

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json",
#     }

#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, json=payload, headers=headers) as response:
#             result = await response.json()
#             return {"run_id": result.get("run_id")}  # Return dictionary format
        
        
# async def get_job_status(databricks_instance, token, run_id):
#     """Checks the status of a Databricks job using its run ID."""
#     url = f"{databricks_instance}/api/2.0/jobs/runs/get?run_id={run_id}"
#     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

#     async with aiohttp.ClientSession() as session:
#         async with session.get(url, headers=headers) as response:
#             return await response.json()
        
        
# async def execute_all(databricks_instance, token, notebook_path, test_cases):
#     """Runs all jobs, waits for completion, and retrieves results."""
    
#     # Step 1: Start all jobs in parallel
#     run_id_map = await execute_parallel(databricks_instance, token, notebook_path, test_cases)
    
#     # Step 2: Wait for all jobs to complete
#     completed_jobs = await wait_for_jobs(databricks_instance, token, run_id_map)
    
#     # Step 3: Fetch results for completed jobs
#     results = {}
#     for car_number, run_id in completed_jobs.items():
#         results[car_number] = await get_job_result(databricks_instance, token, run_id)

#     return results

        
# async def get_job_result(databricks_instance, token, run_id):
#     """Fetches the final result of a completed Databricks job."""
#     url = f"{databricks_instance}/api/2.0/jobs/runs/get-output?run_id={run_id}"
#     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

#     async with aiohttp.ClientSession() as session:
#         async with session.get(url, headers=headers) as response:
#             result = await response.json()
#             return result.get("notebook_output", {}).get("result")  # Extract output
        
        
# async def wait_for_jobs(databricks_instance, token, run_id_map):
#     """Waits for all Databricks jobs to finish."""
#     completed_jobs = {}
    
#     while run_id_map:
#         for car_number, run_id in list(run_id_map.items()):
#             job_status = await get_job_status(databricks_instance, token, run_id)
#             state = job_status["state"]["life_cycle_state"]

#             if state == "TERMINATED":
#                 print(f"Job {run_id} for Car {car_number} is DONE!")
#                 completed_jobs[car_number] = run_id
#                 del run_id_map[car_number]  # Remove finished job
#             elif state == "FAILED":
#                 print(f"Job {run_id} for Car {car_number} FAILED!")
#                 del run_id_map[car_number]  # Remove failed job

#         if run_id_map:  # If there are still jobs running, wait 5s
#             await asyncio.sleep(5)

#     return completed_jobs


# # Define test cases (same set for every benchmark)
TEST_CARS = [
    # ("BNSF", "238515"),
    # ("BNSF", "238571"),
    # ("BNSF", "238804"),
    # ("BNSF", "238805"),
    # ("BNSF", "238864"),
    # ("BNSF", "270598"),
    # ("BNSF", "270647"),
    # ("BNSF", "270846"),
    # ("DTTX", "459385"),
    # ("DTTX", "62579"),
    # ("DTTX", "723873"),
    # ("DTTX", "73030"),
    # ("DTTX", "747095"),
    # ("DTTX", "748863"),
    # ("DTTX", "760434"),
    # ("DTTX", "767861"),
    # ("BNSF", "255125"),
    # ("BNSF", "255236"),
    # ("DTTX", "723755"),
    # ("DTTX", "766537"),
    # ("FEC", "72191"),
    # ("UCRY", "57122"),
    # ("BNSF", "238933"),
    # ("BNSF", "239425"),
    # ("DTTX", "61505"),
    # ("DTTX", "741316"),
    # ("DTTX", "743399"),
    # ("DTTX", "747295"),
    # ("DTTX", "750869"),
    # ("SFLC", "9303"),
    # ("AOK", "74653"),
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

# NUM_RUNS = 1  # Number of times to query each car
# csv_filename = "new_benchmark_results.csv"  # Databricks FileStore path

# # Check if file exists (to avoid overwriting previous runs)
# file_exists = os.path.exists(csv_filename)

# loop = asyncio.get_event_loop()
# results = loop.run_until_complete(
#     execute_all(databricks_instance, databricks_token, db_notebook_path, TEST_CARS)
# )

# # Print the results for debugging
# for car in TEST_CARS:
#     res = results[car]

#     start_time = time.time
        
#     try:
#         ntb_results = json.loads(res["notebook_output"]["result"].strip("'"))
#         train_data = ntb_results["data"]
#         timing_data = ntb_results["timings"]
        
#         # Extract relevant fields
#         trn_id = train_data.get("trn_id", "N/A").strip()
#         trn_dst_cty = train_data.get("dest_city_frefrm", "N/A").strip()
#         trn_dest_st = train_data.get("dest_st", "N/A").strip()
#     except KeyError:
#         # Handle missing data or bad JSON responses
#         trn_id, trn_dst_cty, trn_dest_st = "SymbolNotFound", "NA", "NA"
        
#         execution_time = time.time - start_time

#     row = {
#         "car_id": res.split("-")[0],
#         "car_number": res.split("-")[1],
#         "trn_id": trn_id,
#         "trn_dst_cty": trn_dst_cty,
#         "trn_dest_st": trn_dest_st,
#         "total_execution_time": execution_time,  # Total from function app call to final response
#         "remote_api_request_time": timing_data.get("api_request_time", "N/A"),
#         "remote_sql_execution_time": timing_data.get("sql_execution_time", "N/A"),
#         "remote_dataframe_processing_time": timing_data.get("data_processing_time", "N/A"),
#         "remote_output_preparation_time": timing_data.get("output_prep_time", "N/A"),
#     }

#     # Convert row to DataFrame and append it to CSV (without overwriting previous data)
#     pd.DataFrame([row]).to_csv(csv_filename, mode='a', header=not file_exists, index=False)

#     file_exists = True  # After first write, avoid writing headers again

#     # print(f"Run {i+1}/{NUM_RUNS} - Car: {car_id}-{car_number} - Total Time: {execution_time:.2f}s")

# print(f"Benchmarking complete! Results saved to {csv_filename}")

def execute_notebook(car_id, car_number):
    
        
    databricks_instance = "https://adb-2326721360835808.8.azuredatabricks.net"
    notebook_path = "/Workspace/Users/kevin.morales@bnsf.com/optimized_train_detail"
    token = "dapi391e2e21c61adb340d5bdf76d9e361e2"
    
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

NUM_RUNS = 2  # Number of times to query each car
csv_filename = "new_benchmark_results.csv"  # Databricks FileStore path

# Check if file exists (to avoid overwriting previous runs)
file_exists = os.path.exists(csv_filename)

for car_id, car_number in TEST_CARS:
    for i in range(NUM_RUNS):
        start_time = time.time()  # Start full execution timer

        # Execute the notebook and get the result
        notebook_response = execute_notebook(car_id, car_number)

        # End full execution time tracking
        total_execution_time = time.time() - start_time

        try:
            # Parse the JSON response
            notebook_output = json.loads(notebook_response["notebook_output"]["result"].strip("'"))
            train_data = notebook_output["data"]
            timing_data = notebook_output["timings"]

            # Extract relevant fields
            trn_id = train_data.get("trn_id", "N/A").strip()
            trn_dst_cty = train_data.get("dest_city_frefrm", "N/A").strip()
            trn_dest_st = train_data.get("dest_st", "N/A").strip()
        except (KeyError, IndexError, json.JSONDecodeError):
            trn_id, trn_dst_cty, trn_dest_st = "SymbolNotFound", "NA", "NA"
            timing_data = {}

        # Append results to the list
        row = {
            "car_id": car_id,
            "car_number": car_number,
            "trn_id": trn_id,
            "trn_dst_cty": trn_dst_cty,
            "trn_dest_st": trn_dest_st,
            "total_execution_time": total_execution_time,  # Total from function app call to final response
            "remote_api_request_time": timing_data.get("api_request_time", "N/A"),
            "remote_sql_execution_time": timing_data.get("sql_execution_time", "N/A"),
            "remote_dataframe_processing_time": timing_data.get("data_processing_time", "N/A"),
            "remote_output_preparation_time": timing_data.get("output_prep_time", "N/A"),
        }

        # Convert row to DataFrame and append it to CSV (without overwriting previous data)
        pd.DataFrame([row]).to_csv(csv_filename, mode='a', header=not file_exists, index=False)

        file_exists = True  # After first write, avoid writing headers again

        print(f"Run {i+1}/{NUM_RUNS} - Car: {car_id}-{car_number} - Total Time: {total_execution_time:.2f}s")

print(f"Benchmarking complete! Results saved to {csv_filename}")