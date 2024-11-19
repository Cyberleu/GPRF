
from db_utils import *
from plan import *
import yaml
import glob
from sklearn.model_selection import train_test_split

SEED = 1234
REAPEATED = 5
MVSIZE = 3
config_path = "config.yml"

def generate_select(cond, plan, item, selects) :
    matches = re.findall(r'[a-z_]+[1-9]*\.[a-z]*', cond)
    for match in matches:
        index  = match.find('.')
        table_name = plan.alias_to_table[match[:index]]
        if table_name in item[0]:
            new_select = table_name + '.' + match[index+1:] + ' AS ' + match[:index] + '_' + match[index+1:]
            if new_select not in selects:
                selects.append(new_select)
    return selects  

if __name__ == '__main__':
    sub_plans_set = {}
    with open(config_path, 'r') as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    job_path = d['db_args']['job_path']
    # x_train = glob.glob(job_path + "27c.sql")
    x_train,x_test = train_test_split(glob.glob(job_path + "[0-9]*.sql"),train_size=0.7)
    plans = []
    for i in range(len(x_train)):
        plan = build_and_save_optimizer_plan(x_train[i])
        im = plan.render()
        im.save('/data/homedata/lch/GPRF/im.png')
        plans.append(plan)
        # root_node = plan.get_roots()[0]
        sub_plans = get_sub_plans2(plan)
        for sub_plan in sub_plans:
            tables, alias_tables = get_all_tables(sub_plan)
            plan.sub_alias_tabls.append(alias_tables)
            if tables in sub_plans_set:
                sub_plans_set[tables].append(i)
            else :
                sub_plans_set[tables] = [i]
    # 按照可复用个数进行排序
    sub_plans_set_sorted = sorted(sub_plans_set.items(),key = lambda x:len(x[1]),reverse = True)
    # 生成mv sql语句
    mv_sqls = []
    mv_names = {}
    sql_with_mv = {}
    for item in sub_plans_set_sorted:
        if len(item[0]) < MVSIZE or len(item[1]) < REAPEATED or len(set(item[0])) != len(item[0]):
            continue
        mv_names[item[0]] = f'mv_{len(mv_names)}'
        mv_sql = f'CREATE MATERIALIZED VIEW {mv_names[item[0]]} AS SELECT '
        selects = []
        conditions = {}
        for idx in item[1]:
            if idx in sql_with_mv:
                sql_with_mv[idx].append(item[0])
            else: 
                sql_with_mv[idx] = [item[0]]
            plan = plans[idx]
            for select in plan.query_select:
                selects = generate_select(select['value']['min'], plan, item, selects)
            for cond in plan.query_join_conditions:
                if len(cond['names']) == 1:
                    table_name = plan.alias_to_table[cond['names'][0]]
                    if table_name in item[0]:
                        new_cond = table_name + cond['condition'][len(cond['names'][0]):]
                        if table_name in conditions:
                            conditions[table_name].append(new_cond)
                        else :
                            conditions[table_name] = [new_cond]
                    selects = generate_select(cond['condition'], plan, item, selects)
                else:
                    table_name1 = plan.alias_to_table[cond['names'][0]]
                    table_name2 = plan.alias_to_table[cond['names'][1]]
                    if table_name1 in item[0] and table_name2 not in item[0]:
                        selects = generate_select(cond['condition'], plan, item, selects)
                    elif table_name1 in item[0] and table_name2 not in item[0]:
                        selects = generate_select(cond['condition'], plan, item, selects)
                    elif table_name1 in item[0] and table_name2 in item[0]:
                        index = cond['condition'].find('=')
                        new_cond1 = re.sub(cond['names'][0]+'.',table_name1+'.', cond['condition'][:index]) 
                        new_cond2 = re.sub(cond['names'][1]+'.',table_name2+'.', cond['condition'][index+1:]) 
                        conditions[(table_name1, table_name2)] = [new_cond1 + ' = ' + new_cond2]
        for select in selects:
            mv_sql += select + ' , '
        mv_sql = mv_sql[:-2] + ' FRROM '
        for table in item[0]:
            mv_sql += table + ' , ' 
        mv_sql = mv_sql[:-2] + ' WHERE '
        for key in conditions:
            cond_str = '('
            for cond in conditions[key]:
                cond_str += cond + ' OR '
            cond_str = cond_str[:-3] + ') AND '
            mv_sql += cond_str
        mv_sql = mv_sql[:-4]
        mv_sqls.append(mv_sql)
    # 进行mv替换 
    query_sqls = []
    for idx in sql_with_mv:
        plan = plans[idx]
        replaced_plan = replace_with_mv(plan, sql_with_mv[idx], mv_names)
        sql = generate_sql(replaced_plan)
        query_sqls.append(sql)
        print(1)
    conn = psycopg2.connect(host=d['db_args']['host'],user = d['db_args']['user'],password = d['db_args']['password'],database = d['db_args']['db'])
    for sql in mv_sqls.append(query_sqls):
        cur = conn.cursor()
        try:
            cur.execute(DB_SETTINGS)
            cur.execute(
                f"{sql}")
            data = cur.fetchall()
        except Exception as e:
            cur.close()
            db_rollback(conn)
            raise e
        cur.close()
    

