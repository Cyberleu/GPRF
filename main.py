
from db_utils import *
import yaml
import glob
from sklearn.model_selection import train_test_split

SEED = 1234
config_path = "config.yml"

if __name__ == '__main__':
    with open(config_path, 'r') as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    job_path = d['db_args']['job_path']
    x_train,x_test = train_test_split(glob.glob(job_path + "[0-9]*.sql"))
    for each in x_train:
        with open(each, 'r') as file:
            q = file.read()
        query_tables, reverse_aliases_dict, query_conditions = parse_sql_query(q)
        print(1)