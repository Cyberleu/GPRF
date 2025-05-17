import json
import yaml
import psycopg2
import glob

config_path = "./config/config.yml"
env_path = "./config/postgres_env_config.json"
offical_path = "./config/official.json"
d = {}
env_config = {}


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

with open(config_path, 'r') as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
with open(env_path, "r") as f:
        env_config = json.load(f)
conn = psycopg2.connect(host = d['db_args']['host'],user = d['db_args']['user'],password = d['db_args']['password'],database = d['db_args']['db'])

print_keys = ['']

def print_config():
    for first_key in d:
         for second_key in d[first_key]:
              if second_key.endwith('path'):
                   continue
              

