import os
from datetime import datetime
import pytz

def format_dt_times(utc_datetime, result_timezone="Mountain"):
    # Ensure the input is a timezone-aware datetime object in UTC
    if (
        not isinstance(utc_datetime, datetime)
        or utc_datetime.tzinfo is None
        or utc_datetime.tzinfo.utcoffset(utc_datetime) is None
    ):
        raise ValueError("The input must be a timezone-aware datetime \
            object in UTC.")

    # Define the timezones
    timezones = {
        "Mountain": pytz.timezone("US/Mountain"),
        "Central": pytz.timezone("US/Central"),
        "UTC": pytz.utc,
    }

    # Get the result timezone
    result_tz = timezones.get(result_timezone, pytz.utc)

    # Convert the time to the result timezone
    local_time = utc_datetime.astimezone(result_tz)

    # Return the date and time as a list in 24-hour format
    if result_timezone == "Mountain":
        return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    else:
        return local_time.strftime("%Y-%m-%d %H:%M:%S")
    
def get_seconds_to_look_back():
    """Returns SECONDS_TO_LK_BACK as an int."""
    return int(os.getenv("SECONDS_TO_LK_BACK", "3600"))  # fallback to 1 hour if unset