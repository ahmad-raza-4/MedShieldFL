import os
import subprocess

# List of directories to process
directories = ["1.0", "10.0", "20.0", "30.0"]

# Command to execute
command = [
    "nohup python client_1.py > output_1.log 2>&1 &",
    "nohup python client_2.py >> output_2.log 2>&1 &",
    "nohup python client_3.py >> output_3.log 2>&1 &",
    "nohup python client_4.py >> output_4.log 2>&1 &",
    "nohup python client_5.py >> output_5.log 2>&1 &",
    "nohup python client_6.py >> output_6.log 2>&1 &",
    "nohup python server.py >> server.log 2>&1 &"
]

# Base directory path
base_dir = "/home/dgxuser16/NTL/mccarthy/ahmad/github/adaptive_privacy_fl/adaptive_privacy/alzheimer's/new/"

# Function to run the command in a directory
def run_command_in_dir(directory):
    try:
        # Change to the target directory
        os.chdir(os.path.join(base_dir, directory))
        
        # Execute each command sequentially
        for cmd in command:
            process = subprocess.Popen(cmd, shell=True)
            process.wait()  # Wait for the current command to finish
            
    except Exception as e:
        print(f"Error encountered in directory {directory}: {e}")
        # Continue to the next directory on error

# Iterate over directories and execute the commands
for directory in directories:
    run_command_in_dir(directory)
