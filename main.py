from gprf import GPRF
from db_utils import *
import yaml
import glob
import json
import random
import config


SEED = 1234
INIT = True


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
    
def random_batch_splitter(sql_list, batch_size):
    """
    将给定的SQL语句列表按照指定的batch_size随机分割成多个子列表。

    Args:
        sql_list (list): 包含SQL语句的列表。
        batch_size (int): 每一批的大小。

    Returns:
        list: 包含随机排序并按batch_size分组后的SQL语句的列表。
    """
    if not sql_list or batch_size <= 0:
        return []

    # 创建一个副本以避免修改原列表
    shuffled_sql = sql_list.copy()
    random.shuffle(shuffled_sql)

    batches = []
    for i in range(0, len(shuffled_sql), batch_size):
        batches.append(shuffled_sql[i:i+batch_size])

    return batches

if __name__ == '__main__':
    # 读取训练文件
    sql_list = []
    sql_informs = {}
    job_train_path = config.d['sys_args']['job_train_path']
    x_train = glob.glob(job_train_path + "[0-9]*.sql")
    for each in x_train:
        with open(each, 'r') as file:
            q = file.read()
            sql_list.append(q)
    if(INIT):
        conn = config.conn
        # 获取原始sql的基准信息
        for sql in sql_list:
            query_tables, reverse_aliases_dict, query_conditions, query_select = parse_sql_query(q)
            time = get_cost_from_db(q, conn, False)
            sql_informs[q] = time
        with open(config.d['sys_args']['baseline_path'], 'w') as file:
            json.dump(sql_informs, file, indent=4)
    # 读取相关文件
    baseline = read_json_file(config.d['sys_args']['baseline_path'])
    # 模拟实际情况，sql按批到来
    batches = random_batch_splitter(sql_list, config.d['sys_args']['sql_batch_size'])
    alg = GPRF(baseline, batches)
    alg.run()
    
    

    
    

    
