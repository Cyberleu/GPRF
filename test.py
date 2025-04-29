# # def split_file_into_lines(input_path, output_dir):
# #     import os

# #     try:
# #         # 打开输入文件
# #         with open(input_path, 'r', encoding='utf-8') as file:
# #             lines = file.readlines()

# #         # 创建输出目录如果它不存在
# #         if not os.path.exists(output_dir):
# #             os.makedirs(output_dir)

# #         # 逐行处理
# #         for line_number, line in enumerate(lines, start=1):
# #             filename = f"{line_number}.sql"
# #             output_path = os.path.join(output_dir, filename)

# #             # 写入每一行到新文件中
# #             with open(output_path, 'w', encoding='utf-8') as new_file:
# #                 new_file.write(line.strip())  # 去除换行符

# #         print(f"成功将文件拆分为{len(lines)}个部分，保存在{output_dir}目录下。")

# #     except FileNotFoundError:
# #         print("错误：输入文件未找到。")
# #     except PermissionError:
# #         print("错误：缺少写入权限。")
# #     except Exception as e:
# #         print(f"发生了一个错误：{e}")

# # # 示例用法
# # input_path = "/data/homedata/lch/GPRF/data/job_test.txt"
# # output_dir = "/data/homedata/lch/GPRF/data/test_data"

# # split_file_into_lines(input_path, output_dir)
# """
# 多文本文件词云生成器
# 功能：批量读取txt文件 → 中文分词 → 词频统计 → 生成词云
# 环境需求：Python 3.6+，需要安装以下库：
#     pip install jieba wordcloud matplotlib numpy pillow
# """

# import os
# import jieba
# import jieba.analyse
# from collections import Counter
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# # ---------------------- 配置区域 ----------------------
# TXT_DIR = "/data/homedata/lch/2024spring-dase-nlp/lab2/article"        # 存放txt文件的目录路径
# STOPWORDS_PATH = "/data/homedata/lch/2024spring-dase-nlp/lab1/cn_stopwords.txt"  # 停用词文件路径
#          # 中文字体文件路径（推荐使用微软雅黑）
# OUTPUT_IMAGE = "wordcloud.png"  # 输出图片文件名
# MASK_IMAGE = None               # 词云形状蒙版图片路径（None表示矩形）
# # -----------------------------------------------------

# def read_text_files(directory):
#     """读取目录下所有txt文件内容"""
#     all_text = ""
#     file_count = 0
    
#     for filename in os.listdir(directory):
#         if not filename.endswith(".txt"):
#             continue
            
#         file_path = os.path.join(directory, filename)
#         try:
#             # 自动检测文件编码
#             with open(file_path, 'rb') as f:
#                 raw_data = f.read()
#                 encodings = ['utf-8', 'gbk', 'gb18030', 'big5']
#                 for encoding in encodings:
#                     try:
#                         content = raw_data.decode(encoding)
#                         all_text += content + "\n"
#                         file_count += 1
#                         break
#                     except UnicodeDecodeError:
#                         continue
#         except Exception as e:
#             print(f"无法读取文件 {filename}: {str(e)}")
    
#     print(f"成功读取 {file_count} 个文本文件")
#     return all_text.strip()

# def load_stopwords(filepath):
#     """加载停用词表"""
#     stopwords = set()
#     if os.path.exists(filepath):
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     word = line.strip()
#                     if word:
#                         stopwords.add(word)
#             print(f"已加载 {len(stopwords)} 个停用词")
#         except Exception as e:
#             print(f"加载停用词失败: {str(e)}")
#     else:
#         print("未找到停用词文件，跳过停用词过滤")
#     return stopwords

# def process_text(text, stopwords):
#     """文本预处理与分词"""
#     # 启用并行分词模式（加快速度）
#     jieba.enable_parallel(4)
    
#     # 使用精确模式分词
#     words = jieba.lcut(text)
    
#     # 过滤处理
#     filtered_words = []
#     for word in words:
#         word = word.strip()
#         if len(word) < 2:                # 去除单字
#             continue
#         if word in stopwords:            # 去除停用词
#             continue
#         if word.isdigit():               # 去除纯数字
#             continue
#         filtered_words.append(word)
    
#     jieba.disable_parallel()
#     return filtered_words

# def generate_wordcloud(word_freq, mask=None):
#     """生成词云图片"""
#     # 设置词云参数
#     wc = WordCloud(
#         width=1600,             # 图片宽度
#         height=1200,            # 图片高度
#         background_color='white',  # 背景颜色
#         max_words=200,          # 最多显示词数
#         colormap='viridis',     # 配色方案
#         mask=mask,              # 形状蒙版
#         contour_width=1,        # 轮廓线宽
#         contour_color='steelblue'  # 轮廓颜色
#     )

#     # 生成词云
#     wc.generate_from_frequencies(word_freq)
    
#     # 显示并保存
#     plt.figure(figsize=(20, 15))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
    
#     # 保存图片
#     wc.to_file(OUTPUT_IMAGE)
#     print(f"词云已保存至: {os.path.abspath(OUTPUT_IMAGE)}")

# def main():
#     # 检查输入目录
#     if not os.path.isdir(TXT_DIR):
#         print(f"错误：目录 {TXT_DIR} 不存在！")
#         return

#     # 读取文本内容
#     raw_text = read_text_files(TXT_DIR)
#     if not raw_text:
#         print("错误：未读取到有效文本内容！")
#         return

#     # 加载停用词
#     stopwords = load_stopwords(STOPWORDS_PATH)

#     # 文本处理
#     print("正在分词处理...")
#     words = process_text(raw_text, stopwords)
#     print(f"有效分词数量: {len(words)}")

#     # 词频统计
#     word_counter = Counter(words)
#     top20 = word_counter.most_common(20)
#     print("\nTOP20高频词：")
#     for word, count in top20:
#         print(f"{word}: {count}")

#     # 加载形状蒙版
#     mask = None
#     if MASK_IMAGE and os.path.exists(MASK_IMAGE):
#         try:
#             mask = np.array(Image.open(MASK_IMAGE))
#             print(f"已加载形状蒙版: {MASK_IMAGE}")
#         except Exception as e:
#             print(f"加载蒙版失败: {str(e)}")

#     # 生成词云
#     print("\n生成词云中...")
#     generate_wordcloud(word_counter, mask)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import hashlib

class BinaryTree:
    def __init__(self, root_id):
        self.root = TreeNode(root_id)
        self.available = set()
        self.leaf_nodes = [self.root]
        self.node_map = {root_id: self.root}
    
    def add_child(self, parent_id, child_id):
        parent = self.node_map[parent_id]
        if parent.left and parent.right:
            raise ValueError("Parent already has two children")
        
        child = TreeNode(child_id)
        child.parent = parent
        self.node_map[child_id] = child
        
        if not parent.left:
            parent.left = child
        else:
            parent.right = child
        
        self.leaf_nodes.remove(parent)
        self.leaf_nodes.append(child)
        return True

class TreeNode:
    def __init__(self, node_id):
        self.id = node_id
        self.left = None
        self.right = None
        self.parent = None
    
    def subtree_hash(self):
        """生成子树的唯一哈希标识"""
        def _serialize(node):
            if not node: return 'null'
            return f"({node.id}:{_serialize(node.left)},{_serialize(node.right)})"
        return hashlib.md5(_serialize(self).encode()).hexdigest()

class TreeEnv:
    def __init__(self, M, nodes, max_depth=5):
        self.M = M
        self.nodes = nodes
        self.max_depth = max_depth
        self.action_history = []
        self.current_tree = None
    
    def reset(self, root_id):
        self.current_tree = BinaryTree(root_id)
        self.available = set(self.nodes) - {root_id}
        return self.get_state()
    
    def get_state(self):
        """多维度状态编码"""
        state = np.zeros(len(self.nodes)*3)
        for idx, n in enumerate(self.nodes):
            # 特征1: 节点是否已使用
            state[idx] = 1 if n in self.current_tree.node_map else 0
            # 特征2: 可用连接数
            state[len(self.nodes)+idx] = sum(
                self.M[n][c] for c in self.available
            ) if n in self.current_tree.node_map else 0
            # 特征3: 节点在树中的深度
            state[2*len(self.nodes)+idx] = self._get_depth(n)
        return torch.FloatTensor(state)
    
    def _get_depth(self, node_id):
        depth = 0
        node = self.current_tree.node_map.get(node_id)
        while node and node.parent:
            depth +=1
            node = node.parent
        return depth
    
    def get_legal_actions(self):
        actions = []
        for leaf in self.current_tree.leaf_nodes:
            for candidate in self.available:
                if self.M[leaf.id][candidate]:
                    actions.append((leaf.id, candidate))
        return actions

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.action_scorer = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action_mask):
        features = self.feature_extractor(state)
        scores = self.action_scorer(features)
        # 应用动作掩码
        scores = scores - (1-action_mask)*1e9
        return torch.softmax(scores, dim=0)

class SubtreeOptimizer:
    def __init__(self, M, nodes, batch_size=8, gamma=0.99):
        self.M = M
        self.nodes = nodes
        self.batch_size = batch_size
        self.policy = PolicyNetwork(len(nodes)*3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = gamma
        self.memory = deque(maxlen=1000)
    
    def _get_subtree_rewards(self, batch_trees):
        subtree_counter = defaultdict(int)
        for tree in batch_trees:
            visited = set()
            q = deque([tree.root])
            while q:
                node = q.popleft()
                if node in visited: continue
                visited.add(node)
                # 统计所有非叶子子树
                if node.left and node.right:
                    subtree_counter[node.subtree_hash()] += 1
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
        
        # 计算奖励系数
        total = sum((cnt-1)**2 for cnt in subtree_counter.values() if cnt>1)
        return total / len(batch_trees)
    
    def _build_trajectory(self):
        trajectory = []
        root = np.random.choice(self.nodes)
        env = TreeEnv(self.M, self.nodes)
        state = env.reset(root)
        generated_tree = None  # 初始化树对象
        
        try:
            while True:
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    break
                
                # 生成动作掩码
                action_mask = torch.zeros(len(self.nodes)**2)
                for (p,c) in legal_actions:
                    idx = self.nodes.index(p)*len(self.nodes)+self.nodes.index(c)
                    action_mask[idx] = 1
                
                # 选择动作
                action_probs = self.policy(state, action_mask)
                action_idx = torch.multinomial(action_probs, 1).item()
                parent_idx = action_idx // len(self.nodes)
                child_idx = action_idx % len(self.nodes)
                parent = self.nodes[parent_idx]
                child = self.nodes[child_idx]
                
                # 执行动作
                if env.current_tree.add_child(parent, child):
                    env.available.remove(child)
                    next_state = env.get_state()
                    trajectory.append((state, action_idx, action_mask))
                    state = next_state
                    generated_tree = env.current_tree  # 更新树引用
                else:
                    break
                    
            return trajectory, generated_tree  # 确保返回两个值
            
        except Exception as e:
            print(f"Error during trajectory building: {str(e)}")
            return trajectory, generated_tree  # 即使异常也保持返回结构
    
    def train_batch(self):
        # 生成批量轨迹
        batch_trajectories = []
        batch_trees = []
        
        for _ in range(self.batch_size):
            trajectory, tree = self._build_trajectory()
            batch_trajectories.append(trajectory)
            batch_trees.append(tree)
        
        # 计算公共子树奖励
        R = self._get_subtree_rewards(batch_trees)
        
        # 策略梯度更新
        policy_loss = []
        for trajectory in batch_trajectories:
            returns = 0
            for t in reversed(range(len(trajectory))):
                state, action_idx, action_mask = trajectory[t]
                returns = self.gamma * returns + R
                log_prob = torch.log(self.policy(state, action_mask)[action_idx])
                policy_loss.append(-log_prob * returns)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

# 使用示例
if __name__ == "__main__":
    # 配置参数
    n = 6  # 节点数量
    M = np.random.randint(0, 2, (n,n))  # 随机连接矩阵
    np.fill_diagonal(M, 0)  # 禁止自连接
    nodes = list(range(n))
    
    # 初始化优化器
    optimizer = SubtreeOptimizer(M, nodes, batch_size=8)
    
    # 训练循环
    for epoch in range(1000):
        loss = optimizer.train_batch()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # 生成最优树结构
    final_trees = [optimizer._build_trajectory()[1] for _ in range(5)]
    # 可进一步分析final_trees中的公共子树