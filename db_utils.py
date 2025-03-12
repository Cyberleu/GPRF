import logging
import re
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pathlib import Path
import yaml
import glob
import psycopg2
from moz_sql_parser import parse
import os
import config
import json
from psycopg2 import sql

from plan import SinglePlan

DB_SETTINGS = """BEGIN;
                SET enable_nestloop = off;
                SET join_collapse_limit=20;
                SET from_collapse_limit=20;
                COMMIT;
                """
# SET statement_timeout = 5000;

LOG = logging.getLogger(__name__)


# get original plan from pg 
def build_and_save_optimizer_plan(sql_path):
    conn = config.conn
    with open(sql_path, 'r') as file:
        q = file.read()
    query_tables, reverse_aliases_dict, query_conditions, parsed_select = parse_sql_query(q)
    print(q)
    p = SinglePlan(query_tables, reverse_aliases_dict, query_conditions,parsed_select, q)
    plan, _ = explain_plan_parser(
        p, p.initial_query, conn, exec_time=False)
    return plan


def create_scheme(l_sch):
    '''Creating schema from the list of strings of the kind: `table.column`'''
    res_dict = defaultdict(list)
    for i in l_sch:
        rel, col = i.split('.')
        res_dict[rel].append(col)
    return res_dict


def explain_analyze(sql_query, conn):
    cur = conn.cursor()
    try:
        cur.execute(DB_SETTINGS)
        cur.execute(f"""EXPLAIN ANALYZE {sql_query}""")
        rows = cur.fetchall()
    except Exception as e:
        cur.close()
        db_rollback(conn)
        raise e
    cur.close()
    for row in rows:
        print(row[0])

# 拆分结果格式为name:[t1,t1] cond: "name.col op pred"
def _parse_single_query_condition(qc):
    simple_ops = {
        'eq': '=',
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'neq': '!='
    }
    like_ops = {
        'not_like': 'NOT LIKE',
        'like': 'LIKE'
    }
    exists_ops = {
        'missing': 'IS NULL',
        'exists': 'IS NOT NULL',
    }

    k, v = list(qc.items())[0]
    if k in simple_ops.keys():
        names = []
        names.append(v[0].split('.')[0])

        second_statement = v[1]
        if isinstance(second_statement, dict):
            second_statement = f"'{second_statement['literal']}'"
        elif isinstance(second_statement, str):
            names.append(v[1].split('.')[0])
        elif isinstance(second_statement, (int, float)):
            second_statement = str(second_statement)
        else:
            return None

        cond = f"{v[0]} {simple_ops[k]} {second_statement}"
    elif k == 'in':
        names = [v[0].split('.')[0]]
        if isinstance(v[1]['literal'], str):
            in_part = f"'{v[1]['literal']}'"
        else:
            in_part = ', '.join([f"'{state}'" for state in v[1]['literal']])
        cond = f"{v[0]} in ({in_part})"
    elif k in like_ops.keys():
        names = [v[0].split('.')[0]]
        cond = f"{v[0]} {like_ops[k]} '{v[1]['literal']}'"
    elif k in exists_ops.keys():
        v = [v]
        names = [v[0].split('.')[0]]
        cond = f"{v[0]} {exists_ops[k]}"
    elif k in ['or', 'and']:
        names = []
        cond = []
        for _v in v:
            condition = _parse_single_query_condition(_v)
            names += condition['names']
            cond.append(condition['condition'])
        cond = '(' + f' {k} '.join(cond) + ')'
        names = list(set(names))
    elif k == 'between':
        names = [v[0].split('.')[0]]
        left, right = v[1], v[2]

        if isinstance(left, dict):
            left = v[1]['literal']
        if isinstance(right, dict):
            right = v[2]['literal']

        if isinstance(left, str):
            left = f"'{left}'"
        if isinstance(right, str):
            right = f"'{right}'"

        cond = f"{v[0]} BETWEEN {left} AND {right}"
    else:
        return None

    return {
        'names': names,
        'condition': cond
    }

# 做更细致的拆分，拆分形式见plan.py
def _parse_single_query_condition2(qc):
    simple_ops = {
        'eq': '=',
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'neq': '!='
    }
    like_ops = {
        'not_like': 'NOT LIKE',
        'like': 'LIKE'
    }
    exists_ops = {
        'missing': 'IS NULL',
        'exists': 'IS NOT NULL',
    }

    k, v = list(qc.items())[0]
    cond = {}
    if k in simple_ops.keys():
        names = []
        names.append(v[0].split('.')[0])
        second_statement = v[1]
        if isinstance(second_statement, dict):
            cond['entry_name'] = v[0].split('.')[0]
            cond['col'] = v[0].split('.')[1]
            cond['op'] = simple_ops[k]
            cond['pred'] = f'\'{second_statement["literal"]}\''
        elif isinstance(second_statement, str):
            names.append(v[1].split('.')[0])
            if(len(names) == 2):
                cond['left_entry_name'] = names[0]
                cond['right_entry_name'] = names[1]
                cond['left_col_name'] = v[0].split('.')[1]
                cond['right_col_name'] = v[1].split('.')[1]
            else :
                cond['entry_name'] = v[0].split('.')[0]
                cond['col'] = v[0].split('.')[1]
                cond['op'] = simple_ops[k]
                cond['pred'] = f'\'{second_statement}\''
        elif isinstance(second_statement, (int, float)):
            second_statement = str(second_statement)
            cond['entry_name'] = v[0].split('.')[0]
            cond['col'] = v[0].split('.')[1]
            cond['op'] = simple_ops[k]
            cond['pred'] = second_statement
        else:
            return None

    elif k == 'in':
        names = [v[0].split('.')[0]]
        cond['entry_name'] = v[0].split('.')[0]
        cond['col'] = v[0].split('.')[1]
        cond['op'] = k
        if isinstance(v[1]['literal'], str):
            cond['pred'] = v[1]['literal']
        else:
            in_part = ', '.join([f"'{state}'" for state in v[1]['literal']])
            cond['pred'] = f"({in_part})"
    elif k in like_ops.keys():
        names = [v[0].split('.')[0]]
        cond['entry_name'] = v[0].split('.')[0]
        cond["col"] = v[0].split('.')[1]
        cond["op"] = k
        cond["pred"] = f'\'{v[1]["literal"]}\''
    elif k in exists_ops.keys():
        v = [v]
        names = [v[0].split('.')[0]]
        cond['entry_name'] = v[0].split('.')[0]
        cond["col"] = v[0].split('.')[1]
        cond["op"] = 'IS'
        cond["pred"] = exists_ops[k][3:]
    elif k in ['or', 'and']:
        names = []
        cond = []
        for _v in v:
            condition = _parse_single_query_condition(_v)
            names += condition['names']
            cond.append(condition['condition'])
        cond = '(' + f' {k} '.join(cond) + ')'
        names = list(set(names))
    elif k == 'between':
        names = [v[0].split('.')[0]]
        left, right = v[1], v[2]

        if isinstance(left, dict):
            left = v[1]['literal']
        if isinstance(right, dict):
            right = v[2]['literal']

        if isinstance(left, str):
            left = f"'{left}'"
        if isinstance(right, str):
            right = f"'{right}'"

        cond['entry_name'] = v[0].split('.')[0]
        cond["col"] = v[0].split('.')[1]
        cond["op"] = k
        cond["pred"] = f"{left} AND {right}"
    else:
        return None

    return {
        'names': names,
        'condition': cond
    }


def _get_query_condition(qc,flag):
    conditions = []

    def _get_query_condition_internal(qc, conditions):
        if isinstance(qc, list):
            for _qc in qc:
                _get_query_condition_internal(_qc, conditions)

        elif isinstance(qc, dict):
            if(flag == 1):
                single_cond = _parse_single_query_condition(qc)
            elif(flag == 2):
                single_cond = _parse_single_query_condition2(qc)
            conditions.append(single_cond)
        else:
            raise Exception(
                f"Not supported type {type(qc)}, must be list or dict")

    _get_query_condition_internal(qc, conditions)
    return conditions


def parse_sql_query(sql_query):
    parsed = parse(sql_query)
    if isinstance(parsed['select'], list):
        query_select = [p for p in parsed['select']]
    else:
        query_select = [parsed['select']]
    query_tables = [p['value'] for p in parsed['from']]
    reverse_aliases_dict = {p['name']: p['value'] for p in parsed['from']}
    query_conditions = _get_query_condition(
        parsed['where'].get('and', parsed['where']), 2)
    
    return query_tables, reverse_aliases_dict, query_conditions, query_select, 


def db_rollback(conn):
    curs = conn.cursor()
    curs.execute("ROLLBACK")
    conn.commit()
    curs.close()


def get_cost_from_db(sql_query, conn, is_view = False, exec_time=False, time_limit = 300000, baseline_cost = None, baseline_time = None):
    """
    DEPRECATED
    sql_query: str
        valid sql query
    conn: psycopg2.extensions.connection
        example: conn = config.conn(
            "postgres://imdb:pwdpwd@127.0.0.1:5432/imdb")

    Returns
    -------
    float: Cost
    """
    cur = conn.cursor()
    try:
        cur.execute(DB_SETTINGS)
        # 对于物化视图生成的语句不设置时间限制
        if not is_view:
            cur.execute(f"""SET statement_timeout = {time_limit};COMMIT; """)
        cur.execute(f"""
        EXPLAIN (FORMAT JSON)
        {sql_query}
        """)
        cost_rows = cur.fetchall()
    except Exception as e:
        cur.close()
        db_rollback(conn)
        # 超时时的操作
        return 2 * baseline_cost, 2 * baseline_time


    if exec_time:
        try:
            cur.execute(f"""
            EXPLAIN (FORMAT JSON , ANALYZE)
            {sql_query}
            """)
            time_rows = cur.fetchall()
        except Exception as e:
            cur.close()
            db_rollback(conn)
            return 2 * baseline_cost, 2 * baseline_time
        
    cur.close()
    cost = cost_rows[0][0][0]['Plan']['Total Cost']
    if exec_time:
        time = time_rows[0][0][0]['Plan']['Actual Total Time']
        return cost, time
    return cost, None
    


def parse_explain_json_to_list_of_nodes(
        plan,
        index_cond_names=['Index Cond', 'Recheck Cond'],
        join_cond_names=['Join Filter', 'Hash Cond', 'Merge Cond']):
    # possible_node_types = [
    #     'Hash',
    #     'Materialize',
    #     'Gather',
    #     'Sort',
    #     'Aggregate',
    #     'Partial Aggregate',
    #     'Vector Aggregate',
    #     'Vector Streaming (type: GATHER)',
    #     'Vector Streaming(type: REDISTRIBUTE)',
    #     'Vector Streaming(type: BROADCAST)',
    #     'Vector Streaming(type: PART REDISTRIBUTE PART ROUNDROBIN)',
    #     'Vector Streaming(type: PART LOCAL PART BROADCAST)',
    #     'Vector Materialize',
    #     'Vector Sort',
    # ] + join_types

    join_types = {
        'Hash Join': "HashJoin",
        'Merge Join': "MergeJoin",
        'Nested Loop': "NestLoop",
        'Vector Sonic Hash Join': "HashJoin",
        'Vector Nest Loop': "NestLoop",
        'Vector Merge Join': "MergeJoin",
    }
    same_filter_types = ['Filter']

    dict_of_nodes = {}
    list_of_joins = []
    conds = []
    alias_to_table = {}
    tables = []

    def _extract_index_conditions(_plan, tab_name):
        for cond_name in index_cond_names + same_filter_types:
            if cond_name in _plan:
                cond = _plan[cond_name]
                conds.append({'names': [tab_name], 'condition': cond})

    def _extract_filter_conditions(_plan):
        # find conditions with 2 tables
        for cond_name in join_cond_names + same_filter_types:
            if cond_name in _plan:
                cond = _plan[cond_name]
                results = re.findall(
                    r'([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)( [\=\>\<]+ )+([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)',
                    cond
                )
                for tab1, col1, op, tab2, col2 in results:
                    condition = f'{tab1}.{col1}{op}{tab2}.{col2}'
                    _plan[cond_name] = condition
                    query_condition = {
                        'names': [tab1, tab2],
                        'condition': f'{tab1}.{col1}{op}{tab2}.{col2}',
                    }
                    if query_condition not in conds:
                        conds.append(query_condition)

    def _extract_table_name(_plan):
        if 'Relation Name' in _plan:
            tab_name = _plan['Relation Name']
            if tab_name not in tables:
                tables.append(tab_name)
            if 'Alias' in _plan:
                alias_to_table[_plan['Alias']] = tab_name
                tab_name = _plan['Alias']
            return tab_name

    def _extract_order_and_join_info(plan):
        tabs = []
        if 'Plans' in plan:
            join_nodes = []
            for _plan in plan['Plans']:
                _tabs = _extract_order_and_join_info(_plan)
                join_nodes.append(_tabs)
                tabs.extend(_tabs)
            if len(plan['Plans']) >= 2:
                list_of_joins.append((tuple(tabs), join_nodes))
        if 'Relation Name' in plan:
            alias = _extract_table_name(plan)
            tabs.append(alias)
        tabs = tuple(tabs)
        _extract_filter_conditions(plan)
        if len(tabs) == 1:
            _extract_index_conditions(plan, tabs[0])
        if plan['Node Type'] in join_types:
            tmp_plan = deepcopy(plan)
            del tmp_plan['Plans']
            tmp_plan["table_entries"] = set(tabs)
            tmp_plan["Node Type"] = join_types[tmp_plan["Node Type"]]
            dict_of_nodes[tabs] = tmp_plan
        return tabs

    _extract_order_and_join_info(plan)
    alias_to_table = dict(sorted(alias_to_table.items()))
    labels = {(n,): i for i, n in enumerate(alias_to_table.keys())}
    relabeled_join_list = []
    for p, (c1, c2) in list_of_joins:
        labels[p] = len(labels)
        dict_of_nodes[labels[p]] = dict_of_nodes.pop(p)
        relabeled_join_list.append((labels[p], (labels[c1], labels[c2])))
    return relabeled_join_list, dict_of_nodes, conds, alias_to_table, tables


def get_cost_plan(p, conn, db, exec_time=False):
    """
    Parameters
    ----------
    p: plan.Plan
    conn: psycopg2.extensions.connection
    db: str (db name)
    exec_time: bool, optional
        estimate time if True

    Returns
    -------
    float
        Actual total cost or Actual total execution time if exec_time=True
    """
    cur = conn.cursor()
    query = p.sql_query()
    try:
        cur.execute(DB_SETTINGS)
        cur.execute(
            f"EXPLAIN (FORMAT JSON{', ANALYZE' if exec_time else ''}) {query}")
        data = cur.fetchall()
        cur.close()
    except Exception as e:
        cur.close()
        db_rollback(conn)
        raise e
    index_cond_names = ['Index Cond', 'Recheck Cond']
    join_cond_names = ['Join Filter', 'Hash Cond', 'Merge Cond']
    query_plan = data[0][0][0]['Plan']
    list_of_joins, joins_info, _, _, _ = parse_explain_json_to_list_of_nodes(
        query_plan, index_cond_names, join_cond_names
    )

    plan = SinglePlan(p.query_tables, p.alias_to_table,
                p.query_join_conditions, p.initial_query)
    for node, (c1, c2) in list_of_joins:
        new_node = plan.join(c1, c2)
        if new_node != node:
            raise RuntimeError("Wrong plan parsing")
    mapping = p.check_isomorphism(plan)
    if mapping is not None:
        for n1, n2 in mapping.items():
            assert p.G.nodes[n1]["table_entries"] == plan.G.nodes[n2]["table_entries"]
            if n2 in joins_info:
                join_info = joins_info[n2]
                info_dict = {
                    'cost': join_info['Total Cost'],
                    'time': join_info.get('Actual Total Time') or join_info.get('Actual Max Total Time'),
                    'rows': join_info.get('Actual Rows'),
                    'join_type': join_info.get('Node Type')
                }
                p.G.nodes[n1].update(info_dict)
    else:
        p.G.nodes[p.get_roots()[0]].update(joins_info[plan.get_roots()[0]])
        LOG.error(
            "The tree structures of the original plan and the executed plan are different!")
    cost = query_plan.get(
        'Actual Total Time') if exec_time else query_plan.get('Total Cost')
    return cost


def explain_plan_parser(plan, query, conn, exec_time=False):
    """
    Parameters
    ----------
    plan: plan.Plan
    query: str
        sql query
    conn: psycopg2.extensions.connection
    exec_time: bool, optional
        estimate time if True

    Returns
    -------
    plan.Plan
    float
        Actual total cost
    float
        Actual total execution time if exec_time=True

    WARNING: can change left/right join order.
    """
    cur = conn.cursor()
    try:
        cur.execute(DB_SETTINGS)
        cur.execute(
            f"EXPLAIN (FORMAT JSON{', ANALYZE' if exec_time else ''}) {query}")
        data = cur.fetchall()
    except Exception as e:
        cur.close()
        db_rollback(conn)
        raise e
    cur.close()
    index_cond_names = ['Index Cond', 'Recheck Cond']
    join_cond_names = ['Join Filter', 'Hash Cond', 'Merge Cond']
    query_plan = data[0][0][0]['Plan']
    list_of_joins, joins_info, _, alias_to_table, _ = parse_explain_json_to_list_of_nodes(
        query_plan, index_cond_names, join_cond_names
    )
    p = SinglePlan(plan.query_tables, plan.alias_to_table,
             plan.query_join_conditions, plan.query_select, plan.initial_query)
    for node, (c1, c2) in list_of_joins:
        join_info = joins_info[node]
        info_dict = {
            'cost': join_info['Total Cost'],
            'time': join_info.get('Actual Total Time') or join_info.get('Actual Max Total Time'),
            'rows': join_info.get('Actual Rows'),
            'join_type': join_info.get('Node Type')
        }
        p.join(c1, c2, **info_dict)
    if exec_time:
        return p, query_plan.get('Actual Total Time')
    else:
        return p, query_plan.get('Total Cost')


PSQL_NUMERIC_DTYPES = [
    'smallint', 'integer', 'bigint', 'decimal', 'numeric',
    'real', 'double precision', 'smallserial', 'serial', 'bigserial'
]


def get_column_dtype(conn, table_name, column_name):
    cur = conn.cursor()
    cur.execute(
        f"SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = '{table_name}' and COLUMN_NAME = '{column_name}'")
    data = cur.fetchone()
    cur.close()
    return data[0]


def get_min_max_of_column(conn, table_name, column_name):
    cur = conn.cursor()
    cur.execute(
        f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name}")
    data = cur.fetchone()
    cur.close()
    return data


def scheme_to_minmax_scheme(conn, scheme):
    minmax_scheme = OrderedDict()
    for table_name, columns in scheme.items():
        minmax_scheme[table_name] = OrderedDict()
        for column in columns:
            if get_column_dtype(conn, table_name, column) in PSQL_NUMERIC_DTYPES:
                min_val, max_val = get_min_max_of_column(
                    conn, table_name, column)
                minmax_scheme[table_name][column] = {
                    'numeric': True, 'min': min_val, 'max': max_val}
            else:
                minmax_scheme[table_name][column] = {'numeric': False}
    return minmax_scheme

def drop_all_materialized_views():
    try:
        # 连接数据库
        conn = config.conn
        cursor = conn.cursor()

        # 查询所有物化视图（排除系统视图）
        query = """
            SELECT schemaname, matviewname 
            FROM pg_matviews 
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
        """
        cursor.execute(query)
        views = cursor.fetchall()
        # 生成删除语句
        drop_commands = []
        for schema, view in views:
            drop_commands.append(
                sql.SQL("DROP MATERIALIZED VIEW IF EXISTS {}.{} CASCADE;")
                .format(sql.Identifier(schema), sql.Identifier(view))
            )

        # 执行删除
        for cmd in drop_commands:
            cursor.execute(cmd)
        # 提交事务
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"操作失败: {e}")


def generate_env_config():
    env_config = config.read_json_file(config.offical_path)
    # if os.path.exists(config.env_path):
    #     env_config = config.read_json_file(config.env_path)
    env_config['db_data'] = {}
    sql_list = []
    job_train_path = config.d['sys_args']['job_train_path']
    x_train = glob.glob(job_train_path + "[0-9]*.sql")
    for each in x_train:
        with open(each, 'r') as file:
            q = file.read()
            sql_list.append(q)
        query_tables, reverse_aliases_dict, query_conditions , query_select = parse_sql_query(q)
        query_name = os.path.basename(each)
        cost, time = get_cost_from_db(q, config.conn, exec_time = True, time_limit=3000000)
        env_config['db_data'][query_name] = [query_tables, reverse_aliases_dict, query_conditions, query_select, q,query_name,cost, time]
    with open(config.env_path, "w") as f:
        json.dump(env_config, f)
generate_env_config()
    