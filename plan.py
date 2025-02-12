import io
import json
import re
from collections import defaultdict, deque
from copy import deepcopy
from db_utils import *
from main import d

import networkx as nx
import numpy as np
from PIL import Image

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
            (i, {'name': alias, 'tables': {tab}, 'tab_entries': {alias}})
            for i, (alias, tab) in enumerate(self.alias_to_table.items()))
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
            tables=self.G.nodes[node1]['tables'] | self.G.nodes[node2]['tables'],
            tab_entries=self.G.nodes[node1]['tab_entries'] | self.G.nodes[node2]['tab_entries'],
            **kwargs
        )
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
        left_tables = self.G.nodes[n1]["tab_entries"]
        right_tables = self.G.nodes[n2]["tab_entries"]
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
                    f"Query with CROSS JOIN: {c[0]} : {self.G.nodes[c[0]]['tab_entries']}  join  {c[1]} : {self.G.nodes[c[1]]['tab_entries']}")
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
                l_tables = self.G.nodes[l]['tab_entries']
                r_tables = self.G.nodes[r]['tab_entries']
                return f"({l_subquery} {r_subquery})"

            def _get_join_hint(node):
                hints = []
                for n in self.G:
                    ch = self.G[n]
                    if len(ch) != 0 and 'join_type' in self.G.nodes[n]:
                        join_method = self.G.nodes[n]['join_type'].replace(
                            " ", "").lower()
                        join_tables = ' '.join(self.G.nodes[n]['tab_entries'])
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
        exception_attrs = ['tables', 'tab_entries', 'feature']
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
                alias_tables.append(sorted(plan.G.nodes[edge[1]]['tab_entries'])[0])
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
# 3. cond , scan算子存储形式为：[{"col":"...", "op":"...", "pred":"..."},{""},..]， join/share算子为[{"left_entry_name":"...", "right_entry_name":"...", "left_col_name":"...", "right_col_name":"..."}]
# 连接默认为等值连接，因此不需要记录op 4. share_list, 记录此share算子被哪些plan shared
class GlobalPlan:
    def __init__(self):
        self.G = nx.DiGraph()
        self.roots = [] # 保存每个singlePlan在globalPlan中对应的node id
        self.singlePlans = []
    def merge(self, plan):
        plan = SinglePlan()
        self.singlePlans.append(plan)
        if(len(self.G) == 0):
            self.G = plan.G
            return
        self.G = nx.union(self.G, plan.G, rename = ("Global", "Single"))
        node1, node2 = self.find_shared_op()
        if(node1 == -1):
            # 无法Share，直接加入G, 新加入的plan以‘Plan(index)-’区分
            self.G = nx.union(self.G, plan.G, rename = ("", "Plan{}-".format(len(self.singlePlans))))
            self.roots.add("Plan{}-".format(plan.get_roots[0]))
        else:
            # 合并table上的谓词，join上的不用管
            self.merge_cond(plan.G, node1, node2)
            self.G.nodes[node1]["type"] = "Share"
            self.G.nodes[node1]["share_list"].append(len(self.singlePlans)-1)
            pres = list(plan.G.predecessors(node2))
            # 
            if(len(pres) == 0):
                self.roots.add(node1)
            else:
                self.G = nx.union(self.G, plan.G, rename = ("", "Plan{}-".format(len(self.singlePlans))))
                self.roots.add("Plan{}-".format(plan.get_roots[0]))
                

    # 对于子树覆盖相同表的节点可视为Share算子
    def find_shared_op(self, g2):
        # 越靠后连接的编号越大， 我们要找的是尽可能覆盖多表的
        for i in range(g2.number_of_nodes()-1,-1,-1):
            other_table_entries = g2.nodes[i]["table_entries"]
            for node, node_data in self.G.nodes.items():
                if(node_data["table_entries"] == other_table_entries and len(other_table_entries) >= d['sys_args']['mv_min_size']):
                    return node, i
        return -1, -1
        
    # 合并node1和node2子树中scan算子中的所有cond
    def merge_cond(self, g2, node1, node2):
        assert self.G.nodes[node1]["table_entries"] == g2.nodes[node2]["table_entries"]
        for node_idx, node_data in g2.nodes.items():
            if(node_data["type"] == "Scan"):
                nodes = self.lookup_by_value_within_node(node1, "table_entries", node_data["table_entries"][0])
                self.G.nodes[nodes[0]]["cond"].append(node_data["cond"])
    
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
        

if __name__ == '__main__':
    plan = build_and_save_optimizer_plan("/data/homedata/lch/GPRF/1.sql")
    im = plan.render()
    im.save('/data/homedata/lch/GPRF/im.png')
    print(1)
    