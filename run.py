import os
import shutil
import subprocess
import sys

# Path to the .env file (assumes it's in the same directory as the script)
ENV_FILE = ".env"

# List of service directories
SERVICES = ["LSTM", "Q_and_A", "RAG_service", "sentiment_analysis", "UI"]

# Ask the user for their operating system
print("Please select your operating system:")
print("1. Windows")
print("2. Linux")
print("3. macOS")

os_choice = input("Enter the number corresponding to your operating system (1/2/3): ").strip()

while os_choice not in ["1", "2", "3"]:
    os_choice = input("Please enter 1, 2, or 3: ").strip()

if os_choice == "1":
    user_os = "Windows"
    docker_command_prefix = []
elif os_choice == "2":
    user_os = "Linux"
    docker_command_prefix = ["sudo"]  # Add sudo for Linux
elif os_choice == "3":
    user_os = "macOS"
    docker_command_prefix = []

print(f"You selected: {user_os}")

# Ensure the .env file exists
if not os.path.isfile(ENV_FILE):
    print(f"Error: {ENV_FILE} does not exist!")
    sys.exit(1)

# Allow the user to modify the OPENAI_API_KEY
with open(ENV_FILE, "r") as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    if line.startswith("OPENAI_API_KEY="):
        new_key = input("Enter your OPENAI_API_KEY: ").strip()
        while not new_key:
            new_key = input("Please enter your OPENAI_API_KEY: ").strip()
        line = f"OPENAI_API_KEY={new_key}\n"
    new_lines.append(line)

# Write the updated .env file
with open(ENV_FILE, "w") as file:
    file.writelines(new_lines)

# Copy the .env file to each service directory
print("Copying .env to service directories...")
for service in SERVICES:
    service_path = os.path.join(".", service)
    env_file_target = os.path.join(service_path, ".env")
    if os.path.isdir(service_path):
        shutil.copy(ENV_FILE, env_file_target)
        print(f"Copied .env to {env_file_target}")
    else:
        print(f"Warning: Directory {service} not found. Skipping...")

# Build Docker images
print("\nBuilding Docker images...")
try:
    subprocess.run(docker_command_prefix + ["docker", "compose", "build"], check=True)
except FileNotFoundError:
    print("Error: Docker is not installed or not found in PATH!")
    sys.exit(1)
except subprocess.CalledProcessError:
    print("Error: Docker build failed!")
    sys.exit(1)

# Start Docker containers
print("\nStarting Docker containers...")
try:
    subprocess.run(docker_command_prefix + ["docker", "compose", "up"], check=True)
except FileNotFoundError:
    print("Error: Docker is not installed or not found in PATH!")
    sys.exit(1)
except subprocess.CalledProcessError:
    print("Error: Docker compose up failed!")
    sys.exit(1)


