import requests
import time
import os

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
    
def get_databricks_config():
    """Returns Databricks credentials and notebook path."""
    return (
        os.getenv("DATABRICKS_INSTANCE"),
        os.getenv("DATABRICKS_TOKEN"),
        os.getenv("DATABRICKS_NOTEBOOK_PTH"),
    )