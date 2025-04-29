import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
from collections import defaultdict
from typing import List, Dict

class TreeNode:
    """带哈希缓存的树节点"""
    def __init__(self, id: int):
        self.id = id
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.parent: TreeNode = None
        self._hash = None
        
    def subtree_hash(self) -> str:
        """生成唯一子树标识"""
        if self._hash is None:
            serial = self.serialize()
            self._hash = hashlib.md5(serial.encode()).hexdigest()
        return self._hash
    
    def serialize(self) -> str:
        """序列化子树结构"""
        if self.is_leaf:
            return str(self.id)
        return f"({self.id}[{self.left.serialize()},{self.right.serialize()}])"
    
    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class TreeBuilderEnv:
    """树构建环境"""
    def __init__(self, M: np.ndarray, nodes: List[int], max_depth=10):
        self.M = M          # 连接矩阵
        self.nodes = nodes # 可用节点
        self.max_depth = max_depth
        self.reset()
    
    def reset(self) -> Dict:
        """初始化环境状态"""
        self.root = TreeNode(np.random.choice(self.nodes))
        self.used = {self.root.id}
        self.available = [n for n in self.nodes if n not in self.used]
        self.join_nodes = {}  # 记录合并节点
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """生成多维状态表示"""
        state = {
            'current_tree': self.root,
            'available_nodes': self.available.copy(),
            'join_nodes': list(self.join_nodes.values())
        }
        return state
    
    def get_legal_actions(self) -> List[tuple]:
        """生成合法动作列表：(父节点ID, 子节点ID)"""
        actions = []
        # 现有节点（包括合并节点）
        candidates = list(self.used) + list(self.join_nodes.keys())
        for parent in candidates:
            for child in self.available:
                if self._can_connect(parent, child):
                    actions.append((parent, child))
        return actions
    
    def _can_connect(self, parent: int, child: int) -> bool:
        """检查连接合法性"""
        # 普通节点连接规则
        if parent in self.nodes:
            return self.M[parent][child] == 1
        # 合并节点连接规则（可自定义）
        return True
    
    def step(self, action: tuple) -> tuple:
        """执行连接动作"""
        parent_id, child_id = action
        parent = self._get_node(parent_id)
        child = TreeNode(child_id)
        
        # 连接节点
        if parent.left is None:
            parent.left = child
        elif parent.right is None:
            parent.right = child
        else:
            raise ValueError("父节点已满")
        
        # 生成合并节点
        if parent.left and parent.right:
            join_id = f"join_{parent_id}_{child_id}"
            self.join_nodes[join_id] = parent
        
        # 更新状态
        self.used.add(child_id)
        self.available.remove(child_id)
        
        done = len(self.available) == 0 or len(self.get_legal_actions()) == 0
        return self._get_state(), 0, done, {}

    def _get_node(self, node_id: int) -> TreeNode:
        """根据ID获取节点实例"""
        if node_id in self.join_nodes:
            return self.join_nodes[node_id]
        return self.root if node_id == self.root.id else None

class PolicyNetwork(nn.Module):
    """带动作掩码的策略网络"""
    def __init__(self, input_dim: int, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.action_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None):
        x = self.encoder(state)
        scores = self.action_head(x)
        
        # 应用动作掩码
        if action_mask is not None:
            scores = scores - (1 - action_mask.float()) * 1e9
        return torch.softmax(scores, dim=-1)

class SubtreeOptimizer:
    """批量子树优化器"""
    def __init__(self, M: np.ndarray, nodes: List[int], batch_size=8, gamma=0.95):
        self.M = M
        self.nodes = nodes
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 初始化策略网络
        state_dim = len(nodes) * 3  # 节点状态、可用性、深度
        self.policy = PolicyNetwork(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
    def _get_subtree_reward(self, batch_trees: List[TreeNode]) -> float:
        """计算公共子树奖励"""
        counter = defaultdict(int)
        for tree in batch_trees:
            self._count_subtrees(tree.root, counter)
        return sum((cnt-1)**2 for cnt in counter.values() if cnt > 1) / self.batch_size
    
    def _count_subtrees(self, node: TreeNode, counter: Dict[str, int]):
        """递归统计子树"""
        if node is None or node.is_leaf:
            return
        counter[node.subtree_hash()] += 1
        self._count_subtrees(node.left, counter)
        self._count_subtrees(node.right, counter)
    
    def _encode_state(self, state: Dict) -> torch.Tensor:
        """状态编码为张量"""
        node_status = [1 if n in state['available_nodes'] else 0 for n in self.nodes]
        available_conn = [sum(self.M[n][c] for c in self.nodes) for n in self.nodes]
        depth_info = [self._get_depth(n, state['current_tree']) for n in self.nodes]
        return torch.FloatTensor(node_status + available_conn + depth_info)
    
    def _get_depth(self, node_id: int, root: TreeNode) -> int:
        """获取节点深度"""
        def _depth(node: TreeNode, current: int) -> int:
            if node is None: return 0
            if node.id == node_id: return current
            return max(_depth(node.left, current+1), _depth(node.right, current+1))
        return _depth(root, 0)
    
    def train_batch(self) -> float:
        """执行批量训练"""
        batch_trees = []
        trajectories = []
        
        # 生成轨迹
        for _ in range(self.batch_size):
            env = TreeBuilderEnv(self.M, self.nodes)
            state = env.reset()
            trajectory = []
            done = False
            
            while not done:
                legal_actions = env.get_legal_actions()
                if not legal_actions: break
                
                # 生成动作掩码
                action_mask = torch.zeros(len(self.nodes)**2)
                for (p, c) in legal_actions:
                    idx = self.nodes.index(p)*len(self.nodes) + self.nodes.index(c)
                    action_mask[idx] = 1
                
                # 选择动作
                state_tensor = self._encode_state(state)
                action_probs = self.policy(state_tensor, action_mask)
                action_idx = torch.multinomial(action_probs, 1).item()
                parent = self.nodes[action_idx // len(self.nodes)]
                child = self.nodes[action_idx % len(self.nodes)]
                
                # 执行动作
                next_state, reward, done, _ = env.step((parent, child))
                trajectory.append((state_tensor, action_idx, 0))  # 即时奖励为0
                state = next_state
            
            batch_trees.append(env.root)
            trajectories.append(trajectory)
        
        # 计算奖励
        total_reward = self._get_subtree_reward(batch_trees)
        
        # 策略梯度更新
        policy_loss = []
        for traj in trajectories:
            returns = 0
            for t in reversed(range(len(traj))):
                state_tensor, action_idx, _ = traj[t]
                returns = self.gamma * returns + total_reward
                log_prob = torch.log(self.policy(state_tensor)[action_idx])
                policy_loss.append(-log_prob * returns)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

# 使用示例
if __name__ == "__main__":
    # 参数配置
    n = 6  # 节点数量
    M = np.random.randint(0, 2, (n, n))  # 随机连接矩阵
    np.fill_diagonal(M, 0)  # 禁止自连接
    nodes = list(range(n))
    
    # 初始化优化器
    optimizer = SubtreeOptimizer(M, nodes, batch_size=8)
    
    # 训练循环
    for epoch in range(1000):
        loss = optimizer.train_batch()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # 生成最终树结构
    final_trees = [optimizer.train_batch() for _ in range(5)]