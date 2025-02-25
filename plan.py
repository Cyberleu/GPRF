import io
import json
import re
from collections import defaultdict, deque
from copy import deepcopy
import config
# from db_utils import *


import networkx as nx
import numpy as np
from PIL import Image
import torch
from torch_geometric.data import Data

import logging
LOG = logging.getLogger(__name__)



class SinglePlan():
    def __init__(self, query_tables=None, alias_to_table=None, query_join_conditions=None, query_select = None, initial_query=None, *args):
        self.query_tables = query_tables if query_tables else []
        self.alias_to_table = dict(
            sorted(alias_to_table.items())) if alias_to_table else {}
        self.query_join_conditions = query_join_conditions if query_join_conditions else []

        self._query_join_conditions = defaultdict(list)
        for q in self.query_join_conditions:
            self._query_join_conditions[frozenset(
                q["names"])].append(q["condition"])
        self.rel_leaves = {t: i for i, t in enumerate(self.alias_to_table)}
        self.query_select = query_select
        self.G = nx.DiGraph()
        self.G.add_nodes_from(
            (i, {'name': alias, 'tables': {tab}, 'table_entries': {alias}, 'type':'Scan' })
            for i, (alias, tab) in enumerate(self.alias_to_table.items()))
        for node_idx, node_data in self.G.nodes.items():
            conditions = []
            for cond in query_join_conditions:
                if len(cond['names']) == 1 and node_data['name'] in cond['names']:
                    conditions.append(cond['condition'])
            self.G.nodes[node_idx].update({'conds':conditions})
            
        self.roots = set(self.G.nodes)
        self.initial_query = initial_query
        self._joins = []
        self.sub_alias_tabls = []
        self.alias_to_table_after_replaced = {}
        self.mv_names = []
    @property
    def is_complete(self):
        return len(self.roots) == 1

    def get_roots(self):
        return sorted(self.roots)

    def action_to_join(self, action):
        a1, a2 = action
        roots = self.get_roots()
        return roots[a1], roots[a2]

    def join_to_action(self, join):
        j1, j2 = join
        roots = self.get_roots()
        return roots.index(j1), roots.index(j2)

    def set_node_attrs(self, node, attrs, in_place=True):
        if in_place:
            for name, value in attrs:
                self.G.nodes[node][name] = value
        else:
            p = deepcopy(self)
            for name, value in attrs.items():
                p.G.nodes[node][name] = value
            return p

    def __hash__(self):
        return (hash(frozenset(self.query_tables))
                + hash(len(self.roots))
                + hash(nx.weisfeiler_lehman_graph_hash(self.G, edge_attr='child')))

    def check_isomorphism(self, other):
        """Try to build an isomorphism between given plan tree structures.
         Returns node mapping (self -> other) and None if there is no one-to-one mapping.
         The two plan trees are considered isomorphic if they have
         identical tree structures and aliases in the leaves match.
        """
        g1 = self.G
        g2 = other.G
        mapping = {}
        if (len(self.G) != len(other.G)
            or len(self.roots) != len(other.roots)
                or self.rel_leaves.keys() != other.rel_leaves.keys()):
            return None
        for rel, v in self.rel_leaves.items():
            mapping[v] = other.rel_leaves[rel]
        q = deque(list(self.rel_leaves.values()))
        while len(q) > 0:
            n1 = q.popleft()
            n2 = mapping[n1]
            pred1 = list(g1.predecessors(n1))
            pred2 = list(g2.predecessors(n2))
            if len(pred1) != len(pred2):
                return None
            for p1, p2 in zip(pred1, pred2):
                if p1 not in mapping:
                    mapping[p1] = p2
                    q.append(p1)
                if (mapping[p1] != p2 or g1[p1][n1]['child'] != g2[p2][n2]['child']):
                    return None
        if len(np.unique(list(mapping.values()))) != len(g2):
            return False
        return mapping

    def __eq__(self, other):
        if self._query_join_conditions != other._query_join_conditions:
            return False
        mapping = self.check_isomorphism(other)
        if mapping is None:
            return False
        for n1, n2 in mapping.items():
            if self.G.nodes[n1].get('join_type') != other.G.nodes[n2].get('join_type'):
                return False
        return True

    def __lt__(self, other):
        return len(self.G) < len(other.G)

    def join(self, node1, node2, **kwargs):
        new_node = len(self.G)
        self.G.add_node(
            new_node,
            type = "Join",
            tables=self.G.nodes[node1]['tables'] | self.G.nodes[node2]['tables'],
            table_entries=self.G.nodes[node1]['table_entries'] | self.G.nodes[node2]['table_entries'],
            share_list = [],
            select_cols = set(),
            conds_dict = {},
            **kwargs
        )
        # 确定是通过哪两列进行连接
        conds = []
        for cond in self.query_join_conditions:
            if(len(cond['names']) == 1):
                continue
            left_entries = self.G.nodes[node1]['table_entries']
            right_entries = self.G.nodes[node2]['table_entries']
            if(cond['names'][0] in left_entries and cond['names'][1] in right_entries):
                conds.append(cond['condition'])
            elif(cond['names'][0] in right_entries and cond['names'][1] in left_entries):
                new_cond = {}
                new_cond['left_entry_name'] = cond['condition']['right_entry_name']
                new_cond['right_entry_name'] = cond['condition']['left_entry_name']
                new_cond['left_col_name'] = cond['condition']['right_col_name']
                new_cond['right_col_name'] = cond['condition']['left_col_name']
                conds.append(new_cond)
        self.G.nodes[new_node].update({'conds':conds})
        self.G.add_edge(new_node, node1, child='left')
        self.G.add_edge(new_node, node2, child='right')
        self._joins.append(
            [new_node, self.join_to_action((node1, node2))])
        self.roots -= set([node1, node2])
        self.roots.add(new_node)
        self._last_join_node = new_node
        return new_node

    def disjoin(self, *args: int):
        for node in args:
            self.roots.remove(node)
            for c in self.G[node]:
                self.roots.add(c)
            self.G.remove_node(node)

    def is_inner_join(self, n1, n2):
        left_tables = self.G.nodes[n1]["table_entries"]
        right_tables = self.G.nodes[n2]["table_entries"]
        for r1 in left_tables:
            for r2 in right_tables:
                if frozenset((r1, r2)) in self._query_join_conditions:
                    return True
        return False

    def rel_to_node(self, rel):
        node = self.rel_leaves[rel]
        while True:
            try:
                node = next(self.G.predecessors(node))
            except Exception:
                return node

    def sql_query(self, **kwargs):
        query = self._sql_query_with_hints(**kwargs)
        for n in self.G:
            c = list(self.G[n])
            if len(c) > 0 and not self.is_inner_join(*c):
                LOG.warning(
                    f"Query with CROSS JOIN: {c[0]} : {self.G.nodes[c[0]]['table_entries']}  join  {c[1]} : {self.G.nodes[c[1]]['table_entries']}")
        return query

    # join_collapse_limit, from_collapse_limit, geqo_threshold
    # should be GREATER than number of the tables in query for hints to work

    def _sql_query_with_hints(self):
        if self.initial_query:
            def _get_leading(node):
                successors = self.G[node]
                if len(successors) == 0:
                    alias = self.G.nodes[node]['name']
                    return f"{alias}"
                l, r = successors
                l_subquery = _get_leading(l)
                r_subquery = _get_leading(r)
                l_tables = self.G.nodes[l]['table_entries']
                r_tables = self.G.nodes[r]['table_entries']
                return f"({l_subquery} {r_subquery})"

            def _get_join_hint(node):
                hints = []
                for n in self.G:
                    ch = self.G[n]
                    if len(ch) != 0 and 'join_type' in self.G.nodes[n]:
                        join_method = self.G.nodes[n]['join_type'].replace(
                            " ", "").lower()
                        join_tables = ' '.join(self.G.nodes[n]['table_entries'])
                        hints.append(f"{join_method}({join_tables})")
                return " ".join(hints)

            node = next(iter(self.roots))
            leading = f"leading({_get_leading(node)})"
            join_type_hint = _get_join_hint(node)
            r = re.split(r'SELECT|FROM', self.initial_query)
            return f"SELECT /*+ {join_type_hint} {leading} */ {r[1]} FROM {r[-1]}"
        else:
            raise Exception(
                'self.initial_query is None! For generating gaussdb query you have to provide initial_query')

    def render(self, attr=None, tables=False, dpi=80):
        labels = {}
        for node in self.G.nodes:
            attrs = self.G.nodes[node]
            name = attrs.get("name", str(node))
            if tables and name in self.alias_to_table:
                name = self.alias_to_table[name]
            if attr is not None and attr in attrs:
                name = attrs[attr]
            labels[node] = name
        return render(self.G, labels, dpi)

    def save(self, path):
        m = {n: i for i, n in enumerate(self.G.nodes)}
        g = nx.relabel_nodes(self.G, m)
        exception_attrs = ['tables', 'table_entries', 'feature']
        _joins = [(tuple(g[n]), {k: v for k, v in g.nodes[n].items() if k not in exception_attrs})
                  for n in g.nodes() if n >= len(self.query_tables)]
        with open(path, "w") as f:
            json.dump([self.query_tables, self.alias_to_table,
                       self.query_join_conditions, self.initial_query, _joins], f)

    def load(self, path):
        with open(path, "r") as f:
            *init_args, _joins = json.load(f)
        self.__init__(*init_args)
        for j in _joins:
            # temporarily for backward compatibility
            if isinstance(j[-1], dict):
                self.join(*j[0], **j[1])
            else:
                self.join(*j)

    def reset(self):
        """
        drop all joins in graph
        """
        self.__init__(self.query_tables, self.alias_to_table,
                      self.query_join_conditions, self.initial_query)


def render(graph, labels={}, dpi=100):
    """
    children left-to-right order is preserved when drawing because of 'ordering' option
    right order is guaranteed as long as each vertex is numbered higher than its descendants.
    """
    try:
        import pygraphviz
    except ImportError as e:
        raise ImportError(
            "requires pygraphviz " "http://pygraphviz.github.io/") from e
    A = pygraphviz.AGraph(
        name=graph.name,
        strict=True,
        directed=graph.is_directed(),
        ordering='out',
    )
    for n in reversed(sorted(graph)):
        A.add_node(n)
        for ch in graph[n]:
            A.add_edge(n, ch)
    for n, l in labels.items():
        A.get_node(n).attr["label"] = l
    return Image.open(io.BytesIO(
        A.draw(format='png', prog='dot', args=f'-Gdpi={dpi} -Nfontsize=10 -Nfontname=helvetica')
    ))


def get_sup_plans(p):
    """
    Get plans that can be build with joins from p given conditions.
    """
    plans = []
    nodes = list(p.roots)
    pairs = [(i, j) for i in nodes for j in nodes if i != j]
    for (n1, n2) in pairs:
        if p.is_inner_join(n1, n2):
            new_p = deepcopy(p)
            new_p.join(n1, n2)
            plans.append(new_p)
    return plans


def get_sub_plans(plan):
    """
    Get plans from which one can build the plan with joins.
    """
    plans = {}  # keys - set of disjoins, value - corresponding subplan

    def _get_sub_plans(plan, disjoins):
        plans[disjoins] = plan
        for root in plan.roots:
            if root < len(plan.query_tables):
                continue
            new_disj = disjoins.union((root,))
            if new_disj not in plans:
                p_copy = deepcopy(plan)
                p_copy.disjoin(root)
                _get_sub_plans(p_copy, new_disj)
    _get_sub_plans(deepcopy(plan), frozenset())
    return list(plans.values())

# 取得完整的sub_plan,即len(root) = 1
def get_sub_plans2(plan):
    plans = []
    def _get_sub_plans(root_node):
        new_plan = deepcopy(plan)
        new_plan.roots = set({root_node})
        plans.append(new_plan)
        edges = plan.G.edges(root_node)
        for edge in edges:
            print(edge)
            if edge[1] >= len(plan.query_tables):
                _get_sub_plans(edge[1])
    _get_sub_plans(plan.get_roots()[0])
    return plans

def get_all_tables(plan):
    tables = []
    alias_tables = []
    def _get_all_tables(node):
        edges = plan.G.edges(node)
        for edge in edges:
            if edge[1] < len(plan.query_tables):
                tables.append(sorted(plan.G.nodes[edge[1]]['tables'])[0])
                alias_tables.append(sorted(plan.G.nodes[edge[1]]['table_entries'])[0])
            else :
                _get_all_tables(edge[1])
    _get_all_tables(plan.get_roots()[0])
    tables.sort()
    return tuple(tables), tuple(alias_tables)

def replace_with_mv(plan, sql_with_mv, mv_names):
    del_list = []
    replaced_alias_tables = []
    # plan = Plan()
    plan = deepcopy(plan)
    for mv in sql_with_mv:
        # 对query中出现相同表连接进行特殊处理
        for tables in plan.sub_alias_tabls:
            if len(tables) == len(mv):
                flag = True
                for i in range(len(mv)):
                    if plan.alias_to_table[tables[i]] not in mv:
                        flag = False
                if flag:
                    alias_mv = tables
        flag = False
        for alias in alias_mv:
            if alias in replaced_alias_tables:
                flag = True
        if flag:
            continue
        for i in range(len(plan.query_join_conditions)):
            name = plan.query_join_conditions[i]['names']
            cond = plan.query_join_conditions[i]['condition']
            if len(name) == 1:
                alias_name = name[0]
                if  plan.alias_to_table[alias_name] in mv:
                    new_cond = re.sub(r'\b' + re.escape(alias_name) + r'\.', mv_names[mv] + '.' + alias_name + '_', cond)
                    plan.query_join_conditions[i]['condition'] =  new_cond
                    plan.query_join_conditions[i]['names'][0] = mv_names[mv]
            else:
                alias_name1 = name[0]
                alias_name2 = name[1]
                index = cond.find('=')
                if alias_name1 in alias_mv and alias_name2 in alias_mv:
                    del_list.append(i)
                elif alias_name1 in alias_mv:
                    new_cond = re.sub(r'\b' + re.escape(alias_name1) + r'\.', mv_names[mv] + '.' + alias_name1 + '_', cond)
                    plan.query_join_conditions[i]['condition'] =  new_cond
                    new_cond = plan.query_join_conditions[i]['names'][0] = mv_names[mv]
                elif alias_name2 in alias_mv:
                    new_cond = re.sub(r'\b' + re.escape(alias_name2) + r'\.', mv_names[mv] + '.' + alias_name2 + '_', cond)
                    plan.query_join_conditions[i]['condition'] =  new_cond
                    plan.query_join_conditions[i]['names'][0] = mv_names[mv]  
        plan.query_join_conditions = np.delete(plan.query_join_conditions, del_list).tolist()
        # 替换select中的列名
        for i in range(len(plan.query_select)):
            col_name = plan.query_select[i]['value']['min']
            index = col_name.find('.')
            if plan.alias_to_table[col_name[:index]] in mv:
                new_col_name = mv_names[mv] + '.' + col_name[:index] + '_' + col_name[index + 1 :]
                plan.query_select[i]['value']['min'] = new_col_name
        for alias in alias_mv:
            replaced_alias_tables.append(alias)
        for alias in plan.alias_to_table.keys():
            if alias not in alias_mv:
                plan.alias_to_table_after_replaced[alias] = plan.alias_to_table[alias]
        plan.mv_names.append(mv_names[mv])
    return plan

def generate_sql(plan):
    # plan = Plan()
    sql = 'SELECT '
    for select in plan.query_select:
        sql += 'MIN(' + select['value']['min'] + ' ), '
    sql = sql[:-2] + ' FROM '
    for alias in plan.alias_to_table_after_replaced.keys():
        sql += plan.alias_to_table_after_replaced[alias] + ' AS ' + alias + ' , '
    for mv in plan.mv_names:
        sql += mv + ' , '
    sql = sql[:-2] + ' WHERE '
    for cond in plan.query_join_conditions:
        sql += cond['condition'] + ' AND '
    sql = sql[:-4] + ';'
    return sql

# 一个globalplan由若干个singlePlan组成，singlePlan之间可以用Share算子连接
# node中包含的属性：1. type(scan, join, share) 2. table_entries, 孩子中包含的所有表，用于Share算子的替换
# 3. conds , scan算子存储形式为：[{"col":"...", "op":"...", "pred":"..."},{""},..]， join/share算子为[{"left_entry_name":"...", "right_entry_name":"...", "left_col_name":"...", "right_col_name":"..."}]
# 连接默认为等值连接，因此不需要记录op 4. share_list, 记录此share算子被哪些plan shared 5. select_cols, 对于Share算子需要select出其他使用该Share算子的列，形式为set(('alias','col'))
# 6. conds_dict, 在后续sql构造环节，每个原始查询需要知道在Share节点中涉及到哪些谓词，{plan_idx:conds}
class GlobalPlan:
    def __init__(self):
        self.env_config = config.env_config
        self.G = nx.DiGraph()
        self.roots = [] # 保存每个singlePlan在globalPlan中对应的node id
        self.singlePlans = []
    def merge(self, plan):
        self.singlePlans.append(plan)
        plan_idx = len(self.singlePlans)-1
        if(len(self.G) == 0):
            self.G = nx.union(self.G, plan.G, rename = ('', f'{plan_idx}_'))
            self.roots.append(f'{plan_idx}_{plan.get_roots()[0]}')
            return
        node1, node2 = self.find_shared_op(plan.G)
        if(node1 == -1):
            # 无法Share，直接加入G, 新加入的plan以‘Plan(index)-’区分
            self.G = nx.union(self.G, plan.G, rename = ('', f'{plan_idx}_'))
            self.roots.append(f'{plan_idx}_{plan.get_roots()[0]}')
        else:
            # 合并table上的谓词，join上的不用管
            self.merge_cond(plan.G, node1, node2, plan_idx)
            self.merge_select(plan, node1, node2)
            self.G.nodes[node1]["type"] = "Share"
            self.G.nodes[node1]["share_list"].append(len(self.singlePlans)-1)
            pres = list(plan.G.predecessors(node2))
            # 本身即为根节点
            if(len(pres) == 0):
                self.roots.add(node1)
            else:
                # 删除-合并-连接
                succs = list(plan.G.successors(node2))
                plan.G.remove_nodes_from(succs)
                self.G = nx.union(self.G, plan.G, rename = ('', f'{plan_idx}_'))
                self.G.add_edge(f'{plan_idx}_{pres[0]}', node1)
                self.roots.append(f'{plan_idx}_{plan.get_roots()[0]}')

    # 对于子树覆盖相同表的节点可视为Share算子
    def find_shared_op(self, g2):
        # 越靠后连接的编号越大， 我们要找的是尽可能覆盖多表的
        for i in range(g2.number_of_nodes()-1,-1,-1):
            other_table_entries = g2.nodes[i]["table_entries"]
            for node, node_data in self.G.nodes.items():
                if(node_data["table_entries"] == other_table_entries and len(other_table_entries) >= config.d['sys_args']['mv_min_size']):
                    return node, i
        return -1, -1
    
    # 对于共享节点，需要select出新加入plan所需的列
    def merge_select(self, plan, node1, node2):
        # 原sql中select中的内容要带上
        for select in plan.query_select:
            val = list(select['value'].values())[0]
            alias = val.split('.')[0]
            col = val.split('.')[1]
            self.G.nodes[node1]['select_cols'].add((alias, col))
        # 和原sql的连接列要选出来
        pres = list(nx.ancestors(plan.G, node2))
        for node_idx in pres:
            for cond in plan.G.nodes[node_idx]['conds']:
                if cond['left_entry_name'] in plan.G.nodes[node2]['table_entries']:
                    self.G.nodes[node1]['select_cols'].add((cond['left_entry_name'], cond['left_col_name']))
                elif cond['right_entry_name'] in plan.G.nodes[node2]['table_entries']:
                    self.G.nodes[node1]['select_cols'].add((cond['right_entry_name'], cond['right_col_name']))
        # 所有的谓词要带上
        succs = list(nx.descendants(plan.G, node2))
        succs.append(node2)
        for node_idx in succs:
            if plan.G.nodes[node_idx]['type'] == 'Scan':
                for cond in plan.G.nodes[node_idx]['conds']:
                    self.G.nodes[node1]['select_cols'].add((list(plan.G.nodes[node_idx]['table_entries'])[0],cond['col']))

        
    # 合并node1和node2子树中scan算子中的所有cond,注意需要将不同sql对应的cond区分开来
    def merge_cond(self, g2, node1, node2, plan_idx2):
        plan_idx1, plan_idx2 = get_plan_idx(node1), plan_idx2
        conds = []
        succs = list(nx.descendants(g2, node2))
        for node_idx in succs:
            node_data = g2.nodes[node_idx]
            if(node_data["type"] == "Scan" and len(node_data['conds']) > 0):
                nodes = self.lookup_by_value_within_node(node1, "table_entries", node_data["table_entries"])
                self.G.nodes[nodes[0]]["conds"].extend(node_data["conds"])
                conds.extend(node_data['conds'])
        self.G.nodes[node1]['conds_dict'][plan_idx2] = conds
        # 如果是首次合并，则原来的pred也要加进conds_dict
        if self.G.nodes[node1]['type'] != 'Share':
            conds = []
            succs = list(nx.descendants(self.G, node1))
            for node_idx in succs:
                node_data = self.G.nodes[node_idx]
                if(node_data["type"] == "Scan" and len(node_data['conds']) > 0):
                    conds.extend(node_data['conds'])
            self.G.nodes[node1]['conds_dict'][plan_idx1] = conds
    
    # 在指定node的子树中查找key的value为target的node
    def lookup_by_value_within_node(self, node_index, key, target):
        nodes = []
        successors = list(nx.descendants(self.G, node_index))
        subgraph_nodes = [node_index] + successors
        H = self.G.subgraph(subgraph_nodes)
        for node in H.nodes:
            if(H.nodes[node][key] == target):
                nodes.append(node)
        return nodes

    
    # 生成以node_idx作为根节点的sql,Share算子生成的是物化视图的sql，root生成的是被物化视图替换后的sql
    # 命名规则：1. 视图名称，VIEW_ + Share节点在GlobalPlan中的名称 2. 视图列，别名为原alias_ + col_name
    def generate_sql(self, node_idx, is_view, Leading = True):
        if is_view:
            assert self.G.nodes[node_idx]['type'] == 'Share'
        else:
            assert node_idx in self.roots
        sql = ''
        if is_view:
            sql += f'CREATE MATERIALIZED VIEW VIEW_{node_idx} AS '
            select_stmt = ''
            for alias, col in list(self.G.nodes[node_idx]['select_cols']):
                select_stmt += f'{alias}.{col} AS {alias}_{col} , '
            select_stmt = select_stmt[:-2]
            from_stmt = ''
            join_stmt = ''
            pred_stmt = ''
            succs = list(nx.descendants(self.G, node_idx))
            succs.append(node_idx)
            for succ in succs:
                node_data = self.G.nodes[succ]
                if node_data['type'] != 'Scan':
                    for cond in node_data['conds']:
                        join_stmt += f'{cond["left_entry_name"]}.{cond["left_col_name"]} = {cond["right_entry_name"]}.{cond["right_col_name"]} AND '
                else:
                    from_stmt += f'{list(node_data["tables"])[0]} AS {list(node_data["table_entries"])[0]} , '
                    for cond in node_data['conds']:
                        pred_stmt += f'{list(node_data["table_entries"])[0]}.{cond["col"]} {cond["op"]} {cond["pred"]} AND '
            from_stmt = from_stmt[:-2]
            where_stmt = pred_stmt + join_stmt
            where_stmt = where_stmt[:-4]
        elif node_idx in self.roots:
            # share 算子可能会有包含，采用bfs算法取最外层的share算子
            succs = list(nx.bfs_tree(self.G, source=node_idx).nodes())
            # Share算子的root
            share_root_nodes = []
            # View中的所有算子（不包含root）
            share_nodes = []
            for succ in succs:
                if self.G.nodes[succ]['type'] == 'Share':
                    # 判断包含关系
                    pres = list(nx.ancestors(self.G, succ))
                    contained = False
                    for pre in pres:
                        if pre in share_root_nodes:
                            contained = True
                            break
                    if not contained:
                        share_root_nodes.append(succ)
            for node in share_root_nodes:
                share_nodes.extend(list(nx.descendants(self.G, node)))
            # 将原始列映射到需要被替换成的视图
            alias2view = {}
            for node in share_root_nodes:
                for entry_name in self.G.nodes[node]['table_entries']:
                    alias2view[entry_name] = node
            select_stmt = ''
            plan = self.singlePlans[self.roots.index(node_idx)]
            for select in plan.query_select:
                key = list(select['value'].keys())[0]
                alias, col = select['value'][key].split('.')
                if alias in alias2view:
                    col = f'{alias}_{col}'
                    alias = 'VIEW_' + alias2view[alias]
                select_stmt += f'{key}({alias}.{col}) AS {select["name"]} , '
            join_stmt = ''
            pred_stmt = ''
            from_stmt = ''
            succs = list(nx.descendants(self.G, node_idx))
            succs.append(node_idx)
            for node in succs:
                node_data = self.G.nodes[node]
                if node in share_root_nodes:
                    from_stmt += f'VIEW_{node} , '
                    # 把view中属于本plan的pred加上
                    for cond in self.G.nodes[node]['conds_dict'][self.roots.index(node_idx)]:
                        pred_stmt += f'VIEW_{node}.{cond["entry_name"]}_{cond["col"]} {cond["op"]} {cond["pred"]} AND '
                elif node in share_nodes:
                    continue
                elif node_data['type'] == 'Join':
                    a1,a2,c1,c2 = node_data['conds'][0]['left_entry_name'],node_data['conds'][0]['right_entry_name'],node_data['conds'][0]['left_col_name'], node_data['conds'][0]['right_col_name'] 
                    if a1 in alias2view:
                        c1 = f'{a1}_{c1}'
                        a1 = 'VIEW' + alias2view[a1]
                    if a2 in alias2view:
                        c2 = f'{a2}_{c2}'
                        a2 = 'VIEW_' + alias2view[a2]
                    join_stmt += f'{a1}.{c1} = {a2}.{c2} AND '
                elif node_data['type'] == 'Scan':
                    from_stmt += f'{list(node_data["tables"])[0]} AS {list(node_data["table_entries"])[0]} , '
                    for cond in node_data['conds']:
                        pred_stmt += f'{cond["entry_name"]}.{cond["col"]} {cond["op"]} {cond["pred"]} AND '
            select_stmt = select_stmt[:-2]
            from_stmt = from_stmt[:-2]
            where_stmt = pred_stmt + join_stmt
            where_stmt = where_stmt[:-4]
        sql += ' SELECT ' + select_stmt + ' FROM ' + from_stmt + ' WHERE ' + where_stmt + ';'
        def _get_leading(node):
                succs = list(self.G.successors(node))
                if self.G.nodes[node]['type'] == 'Share':
                    return f"VIEW_{node}"
                if len(succs) == 0:
                    alias = self.G.nodes[node]['name']
                    return f"{alias}"
                l, r = succs
                l_subquery = _get_leading(l)
                r_subquery = _get_leading(r)
                return f"({l_subquery} {r_subquery})"
        if Leading == True:
            sql = f"/*+ Leading({_get_leading(node_idx)}) */" + sql
        return sql
    
def get_plan_idx(node):
    parts = node.split('_')
    plan_idx = parts[0]  
    return int(plan_idx)

# 将nx图转为pyg图作为gnn的输入
def convert_nx_to_pyg(G):
    # 获取所有节点并排序，确保顺序一致
    nodes = list(G.nodes())
    node_idx = {node: idx for idx, node in enumerate(nodes)}

    # 生成节点特征（例如，每个节点一个特征值为1）
    n_nodes = len(nodes)
    x = torch.ones(n_nodes, 1, dtype=torch.float32)
    
    # 处理边信息
    edges = list(G.edges())
    sources = [node_idx[e[0]] for e in edges]
    targets = [node_idx[e[1]] for e in edges]

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

        

if __name__ == '__main__':
    # plan1 = build_and_save_optimizer_plan("/data/homedata/lch/GPRF/1.sql")
    # plan2 = build_and_save_optimizer_plan("/data/homedata/lch/GPRF/2.sql")
    # im = plan1.render()
    # im.save('/data/homedata/lch/GPRF/im1.png')
    # im = plan2.render()
    # im.save('/data/homedata/lch/GPRF/im2.png')
    gp = GlobalPlan()
    gp.merge(plan1)
    gp.merge(plan2)
    print(gp.roots)
    shared_node = ''
    for node in gp.G.nodes:
        if gp.G.nodes[node]['type'] == 'Share':
           shared_node = node
    print(gp.generate_sql(shared_node, True)) 
    print(gp.generate_sql(gp.roots[0], False))
    