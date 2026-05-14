# PROJECT DESIGN: Label-Efficient Fraud Detection on Elliptic Bitcoin with GNN + Active Learning

> **Môn học:** Deep Learning for Data Science  
> **Dạng project:** Applied Deep Learning / Graph Machine Learning / Active Learning  
> **Mức độ:** Intermediate-Advanced  
> **Trọng tâm:** Thiết kế một pipeline phát hiện giao dịch Bitcoin bất hợp pháp vừa đúng về mặt thực nghiệm, vừa có khả năng quan sát, kiểm tra, tái lập và mở rộng.

---

## 0. Executive Summary

Dự án xây dựng một hệ thống **Label-Efficient Fraud Detection** trên **Elliptic Bitcoin Dataset** bằng cách kết hợp:

1. **Graph Neural Networks (GNN)** để khai thác quan hệ giao dịch trong graph.
2. **Deep Active Learning (AL)** để chọn những node cần gán nhãn nhất, thay vì gán nhãn ngẫu nhiên.
3. **MLOps-grade pipeline** để theo dõi toàn bộ quá trình train, Active Learning loop, pool state, checkpoint, metrics, artifacts và lỗi dữ liệu.

Câu hỏi nghiên cứu chính:

> Với cùng một annotation budget nhỏ, ví dụ chỉ vài phần trăm node có nhãn được dùng để train, các chiến lược Active Learning như Entropy, Margin, Coreset có giúp GNN phát hiện giao dịch bất hợp pháp tốt hơn Random Sampling hay không?

Luận điểm cần chứng minh:

1. **GNN > MLP:** graph structure giúp ích cho fraud detection.
2. **Active Learning > Random Sampling:** chọn node thông minh giúp đạt hiệu năng tốt hơn với cùng số nhãn.
3. **Label-efficient compact GNN:** dùng ít nhãn hơn, model nhỏ hơn, train hợp lý hơn nhưng vẫn đạt kết quả cạnh tranh.

---

## 1. Scope chính thức của project

### 1.1 Bắt buộc làm

| Hạng mục | Trạng thái | Lý do |
|---|---:|---|
| Elliptic Bitcoin Dataset | Bắt buộc | Dataset thật, phù hợp graph fraud detection |
| Temporal split | Bắt buộc | Tránh leakage nghiêm trọng |
| Unknown node handling | Bắt buộc | 77% node không có nhãn thật, phải xử lý đúng |
| Feature normalization train-only | Bắt buộc | Tránh leak thống kê val/test |
| Inductive graph masking | Bắt buộc | Không để test nodes tham gia message passing khi train |
| MLP baseline | Bắt buộc | Chứng minh graph có ích |
| GCN student | Bắt buộc | Backbone GNN chính, nhỏ, dễ train |
| GraphSAGE student | Nên làm | Backbone so sánh, mạnh hơn cho inductive setting |
| Random / Entropy / Margin / Coreset | Bắt buộc | Bộ AL strategy chính |
| Focal Loss hoặc Weighted CE | Bắt buộc | Xử lý class imbalance |
| Learning curves | Bắt buộc | Bằng chứng chính cho label efficiency |
| AUC-ROC, Average Precision, F1-macro, Recall@K | Bắt buộc | Accuracy không phù hợp dữ liệu mất cân bằng |
| Checkpoint + resume | Bắt buộc | Chạy nhiều vòng AL dễ lỗi, cần khôi phục |
| Experiment tracking | Bắt buộc | Quan sát train và pipeline |
| Error analysis | Bắt buộc | Giải thích khi AL tốt hoặc không tốt |
| README + resource report | Bắt buộc | Phù hợp yêu cầu đồ án và tái lập |

### 1.2 Làm nếu còn thời gian

| Hạng mục | Vai trò |
|---|---|
| Label Propagation baseline | Baseline semi-supervised |
| Wilcoxon test với 10 seeds | Kiểm định thống kê mạnh hơn |
| KD-safe | Chỉ dùng trong phân tích phụ, không làm claim chính |
| Gradio demo | Demo trực quan |
| ONNX/TorchScript export | Bổ sung inference/deployment |

### 1.3 Không đưa vào claim chính

| Hạng mục | Lý do loại khỏi main claim |
|---|---|
| Random split | Sai với temporal graph, gây leakage |
| Teacher full-label train student trong AL | Làm hỏng claim label-efficient |
| Imbalance-aware query dùng true label | Không thực tế, gây label leakage |
| AGE / GAT | Có thể quá rộng cho scope đồ án |
| Accuracy làm metric chính | Không phù hợp vì fraud rất hiếm |

---

## 2. Dataset: Elliptic Bitcoin Dataset

### 2.1 Thông tin cần báo cáo

| Thuộc tính | Giá trị |
|---|---|
| Dataset | Elliptic Bitcoin Dataset |
| Framework | PyTorch Geometric |
| Nodes | ~203,769 transactions |
| Edges | ~234,355 |
| Node features | 166 features/node |
| Labels | illicit, licit, unknown |
| Time steps | 49 |
| Class imbalance | illicit khoảng 2%, licit khoảng 21%, unknown khoảng 77% |

Load cơ bản:

```python
from torch_geometric.datasets import EllipticBitcoinDataset

dataset = EllipticBitcoinDataset(root="data/elliptic")
data = dataset[0]
```

### 2.2 Vì sao dataset phù hợp

- Có sẵn trong PyG, không cần crawling.
- Là dữ liệu giao dịch Bitcoin thực tế.
- Có graph structure rõ ràng để GNN khai thác.
- Có temporal dimension, phù hợp thử nghiệm inductive và leakage-safe evaluation.
- Class imbalance nặng, phù hợp để chứng minh giá trị của Active Learning và loss function chuyên biệt.

---

## 3. Các lỗi leakage cần chặn bằng thiết kế hệ thống

Đây là phần quan trọng nhất của project. Pipeline phải được thiết kế sao cho **không thể vô tình chạy sai split hoặc dùng nhãn sai**.

### 3.1 Temporal leakage

Không dùng random split.

```python
train_time = data.time_step <= 34
val_time = (data.time_step >= 35) & (data.time_step <= 40)
test_time = data.time_step >= 41
```

| Split | Time steps | Vai trò |
|---|---:|---|
| Train pool | 1-34 | Active Learning pool |
| Validation | 35-40 | Early stopping, hyperparameter tuning |
| Test | 41-49 | Final evaluation / fixed test learning curve |

Quy tắc:

- Train chỉ dùng node trong time step 1-34.
- Validation chỉ dùng 35-40.
- Test chỉ dùng 41-49.
- Không dùng test để chọn hyperparameter, chọn checkpoint, chọn AL strategy trong quá trình chạy.

### 3.2 Unknown node handling

Cần phân biệt rõ hai khái niệm:

| Khái niệm | Ý nghĩa |
|---|---|
| Unknown label trong dataset | Node không có nhãn thật, không dùng supervised loss |
| Unlabeled pool trong Active Learning | Node có nhãn thật nhưng bị che nhãn trong mô phỏng AL |

Quy tắc:

```python
known_mask = data.y != UNKNOWN_LABEL
supervised_train_mask = train_time & known_mask
val_mask = val_time & known_mask
test_mask = test_time & known_mask
```

- Unknown nodes vẫn được giữ trong graph để message passing.
- Unknown nodes không được dùng để tính loss.
- Unknown nodes không được đưa vào AL query pool.
- Unknown labels không được dùng để tạo class ratio, sampling target hoặc evaluation.

### 3.3 Feature normalization leakage

Sai:

```python
scaler.fit(data.x)  # leak val/test statistics
```

Đúng:

```python
scaler.fit(data.x[train_time])
data.x = scaler.transform(data.x)
```

Ghi chú:

- Có thể fit scaler trên tất cả train-time nodes, bao gồm known và unknown, vì feature là dữ liệu quan sát được ở giai đoạn train.
- Không fit trên validation/test.
- Không refit scaler sau mỗi AL round.

### 3.4 Graph-level leakage qua message passing

Trong quá trình train, model không được message passing qua test nodes.

Đề xuất dùng 3 graph views:

| Graph view | Nodes trong forward | Dùng khi nào |
|---|---|---|
| `G_train` | time step 1-34 | Train model |
| `G_val` | time step 1-40 | Validation / early stopping |
| `G_test` | time step 1-49 | Final inference, no gradient |

Quy tắc:

- Train loss chỉ tính trên labeled pool hiện tại `L`.
- Validation dùng để early stopping và checkpoint selection.
- Test chỉ dùng để báo cáo, không dùng để tune.

---

## 4. Kiến trúc thư mục đề xuất

```text
fraud-gnn-al/
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   │   └── elliptic.yaml
│   ├── model/
│   │   ├── mlp.yaml
│   │   ├── gcn.yaml
│   │   └── graphsage.yaml
│   ├── al_strategy/
│   │   ├── random.yaml
│   │   ├── entropy.yaml
│   │   ├── margin.yaml
│   │   └── coreset.yaml
│   ├── trainer/
│   │   └── default.yaml
│   ├── tracking/
│   │   └── wandb.yaml
│   └── experiment/
│       ├── debug.yaml
│       └── final.yaml
│
├── src/
│   ├── data_layers/
│   │   ├── base_graph_dataset.py
│   │   ├── graph_contracts.py
│   │   ├── graph_views.py
│   │   ├── splitters/
│   │   │   └── temporal_splitter.py
│   │   └── datasets/
│   │       └── elliptic_dataset.py
│   │
│   ├── model_zoo/
│   │   ├── base_graph_model.py
│   │   └── models/
│   │       ├── mlp_model.py
│   │       ├── gcn_model.py
│   │       └── graphsage_model.py
│   │
│   ├── active_learning/
│   │   ├── base_strategy.py
│   │   ├── pool_manager.py
│   │   └── strategies/
│   │       ├── random_sampling.py
│   │       ├── entropy_sampling.py
│   │       ├── margin_sampling.py
│   │       └── coreset_sampling.py
│   │
│   ├── execution_engine/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── al_orchestrator.py
│   │
│   ├── evaluation/
│   │   ├── metric_manager.py
│   │   ├── checkpoint_manager.py
│   │   ├── learning_curve.py
│   │   └── visualizer.py
│   │
│   ├── observability/
│   │   ├── event_logger.py
│   │   ├── pipeline_monitor.py
│   │   ├── artifact_store.py
│   │   └── experiment_tracker.py
│   │
│   ├── factories/
│   │   ├── dataset_factory.py
│   │   ├── model_factory.py
│   │   └── strategy_factory.py
│   │
│   └── utils/
│       ├── seed_manager.py
│       ├── device.py
│       └── io.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── leakage/
│
├── notebooks/
│   └── analysis.ipynb
│
├── scripts/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   ├── export_results.py
│   └── validate_pipeline.py
│
├── outputs/
├── reports/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 5. Nguyên tắc thiết kế bất biến

| # | Quy tắc |
|---:|---|
| 1 | `data_layers/` và `model_zoo/` không import từ `execution_engine/`. |
| 2 | `execution_engine/` chỉ nhận object qua interface, không biết concrete class cụ thể. |
| 3 | `Trainer` không biết tên dataset, model hoặc AL strategy. |
| 4 | Mọi concrete dataset/model/strategy phải đăng ký bằng `@register_*`. |
| 5 | `configs/` là single source of truth để đổi dataset/model/strategy. |
| 6 | Pool state phải serializable sau mỗi AL round. |
| 7 | Mỗi run phải lưu resolved config, seed, git hash, device info, package versions. |
| 8 | Test set không bao giờ được dùng để early stopping, hyperparameter tuning hoặc chọn strategy. |
| 9 | Unknown labels không bao giờ được dùng làm supervised signal. |
| 10 | Nếu pipeline phát hiện config có nguy cơ leakage, phải fail fast. |

---

## 6. Interfaces cốt lõi

### 6.1 BaseGraphDataset

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch_geometric.data import Data

@dataclass
class GraphDatasetInfo:
    name: str
    version: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    time_steps: int
    label_names: list[str]
    unknown_label: int

class BaseGraphDataset(ABC):
    @abstractmethod
    def load(self, root: str) -> Data:
        ...

    @abstractmethod
    def preprocess(self, data: Data) -> Data:
        ...

    @abstractmethod
    def build_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def build_graph_views(self, data: Data) -> Dict[str, Data]:
        ...

    @abstractmethod
    def get_info(self) -> GraphDatasetInfo:
        ...
```

### 6.2 BaseGraphModel

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
from torch_geometric.data import Data

class BaseGraphModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Return raw logits for all nodes in the graph view."""
        ...

    @abstractmethod
    def loss_compute(self, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute supervised loss only on selected mask."""
        ...

    @abstractmethod
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Return node embeddings before classifier, shape [num_nodes, hidden_dim]."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        ...
```

Lưu ý quan trọng:

- `forward()` trả về logits, không softmax.
- `get_embeddings()` trả về embedding trước classifier để phục vụ Coreset hoặc các strategy diversity-based.
- `loss_compute()` luôn nhận mask để tránh tính loss trên unknown, val/test hoặc unlabeled nodes.

### 6.3 BaseALStrategy

```python
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data

class BaseALStrategy(ABC):
    @abstractmethod
    def score_samples(
        self,
        model,
        graph: Data,
        candidate_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return score for each candidate index. Higher score = more useful to query."""
        ...

    @abstractmethod
    def query_indices(
        self,
        scores: torch.Tensor,
        candidate_indices: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        ...

    def query(self, model, graph, candidate_indices, budget, device):
        scores = self.score_samples(model, graph, candidate_indices, device)
        return self.query_indices(scores, candidate_indices, budget)
```

---

## 7. Factory + Registry Pattern

### 7.1 Dataset factory

```python
_DATASET_REGISTRY = {}

def register_dataset(name: str):
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator

class DatasetFactory:
    @staticmethod
    def create(name: str, **kwargs):
        if name not in _DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_REGISTRY)}")
        return _DATASET_REGISTRY[name](**kwargs)
```

### 7.2 Model factory

```python
_MODEL_REGISTRY = {}

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs):
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY)}")
        return _MODEL_REGISTRY[name](**kwargs)
```

### 7.3 Strategy factory

```python
_STRATEGY_REGISTRY = {}

def register_strategy(name: str):
    def decorator(cls):
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

class StrategyFactory:
    @staticmethod
    def create(name: str, **kwargs):
        if name not in _STRATEGY_REGISTRY:
            raise ValueError(f"Unknown AL strategy: {name}. Available: {list(_STRATEGY_REGISTRY)}")
        return _STRATEGY_REGISTRY[name](**kwargs)
```

---

## 8. Data Pipeline chi tiết

### 8.1 EllipticDataset responsibilities

`EllipticDataset` không chỉ load data. Nó phải chịu trách nhiệm tạo những mask và graph views an toàn.

```python
@register_dataset("elliptic")
class EllipticDataset(BaseGraphDataset):
    def __init__(self, root: str, unknown_label: int = -1, **kwargs):
        self.root = root
        self.unknown_label = unknown_label
        self.data = None
        self.masks = None
        self.graph_views = None
        self.scaler = None

    def load(self, root: str):
        dataset = EllipticBitcoinDataset(root=root)
        self.data = dataset[0]
        return self.data

    def preprocess(self, data):
        self.masks = self.build_masks(data)
        self.scaler = StandardScaler()
        train_time = self.masks["train_time"]
        data.x[train_time] = torch.tensor(
            self.scaler.fit_transform(data.x[train_time].cpu().numpy()),
            dtype=torch.float32,
        )
        non_train = ~train_time
        data.x[non_train] = torch.tensor(
            self.scaler.transform(data.x[non_train].cpu().numpy()),
            dtype=torch.float32,
        )
        self.graph_views = self.build_graph_views(data)
        return data
```

### 8.2 Masks bắt buộc

| Mask | Ý nghĩa |
|---|---|
| `train_time` | node time step 1-34 |
| `val_time` | node time step 35-40 |
| `test_time` | node time step 41-49 |
| `known_mask` | node có nhãn thật |
| `supervised_train_mask` | train_time & known_mask |
| `val_mask` | val_time & known_mask |
| `test_mask` | test_time & known_mask |
| `unknown_mask` | node không có nhãn thật |

### 8.3 Graph views

```python
def build_graph_views(data, masks):
    g_train = subgraph_by_node_mask(data, masks["train_time"])
    g_val = subgraph_by_node_mask(data, masks["train_time"] | masks["val_time"])
    g_test = data  # all nodes, inference only
    return {"train": g_train, "val": g_val, "test": g_test}
```

Quan trọng:

- `g_train` không chứa test nodes.
- `g_val` không chứa test nodes.
- `g_test` chỉ dùng ở `model.eval()` + `torch.no_grad()`.

---

## 9. Pool Manager cho Active Learning

### 9.1 Pool semantics

| Set | Nguồn | Có nhãn thật không? | Model được thấy nhãn không? |
|---|---|---:|---:|
| `L` | supervised_train_mask | Có | Có |
| `U` | supervised_train_mask - L | Có | Chưa, chỉ oracle biết |
| `Val` | val_mask | Có | Chỉ dùng evaluation/early stopping |
| `Test` | test_mask | Có | Chỉ dùng final evaluation |
| Unknown nodes | unknown_mask | Không | Không bao giờ dùng loss/query |

### 9.2 PoolManager

```python
class PoolManager:
    def __init__(self, train_indices, y, seed_size, seed):
        self.y = y
        self.seed = seed
        self.labeled_indices = stratified_initial_sample(train_indices, y, seed_size, seed)
        self.unlabeled_indices = sorted_tensor_difference(train_indices, self.labeled_indices)
        self.history = []

    def update(self, queried_indices, round_id, strategy_name, scores=None):
        self.labeled_indices = torch.cat([self.labeled_indices, queried_indices]).unique()
        self.unlabeled_indices = sorted_tensor_difference(self.unlabeled_indices, queried_indices)
        self.history.append({
            "round": round_id,
            "strategy": strategy_name,
            "queried_indices": queried_indices.cpu().tolist(),
            "num_labeled": int(len(self.labeled_indices)),
            "num_unlabeled": int(len(self.unlabeled_indices)),
        })

    def state_dict(self):
        return {
            "seed": self.seed,
            "labeled_indices": self.labeled_indices.cpu().tolist(),
            "unlabeled_indices": self.unlabeled_indices.cpu().tolist(),
            "history": self.history,
        }
```

### 9.3 Lưu pool state

Sau mỗi round phải lưu:

```text
outputs/{experiment_name}/{run_id}/round_{r}/pool_state.json
```

Nội dung:

```json
{
  "round": 3,
  "seed": 42,
  "strategy": "coreset",
  "labeled_indices": [1, 8, 10],
  "unlabeled_indices": [2, 3, 4],
  "queried_indices": [99, 105],
  "class_ratio_labeled": {"licit": 0.82, "illicit": 0.18},
  "class_ratio_queried": {"licit": 0.75, "illicit": 0.25}
}
```

---

## 10. Models

### 10.1 MLP baseline

Vai trò: baseline không dùng graph.

```text
Linear(166 -> 128)
ReLU
Dropout(0.3)
Linear(128 -> 2)
```

Đầu vào: `data.x`  
Không dùng: `edge_index`

### 10.2 GCN student

Vai trò: model GNN chính.

```text
GCNConv(166 -> 128)
ReLU
Dropout(0.3)
GCNConv(128 -> 2)
```

### 10.3 GraphSAGE student

Vai trò: backbone thứ hai, phù hợp inductive setting.

```text
SAGEConv(166 -> 128)
ReLU
Dropout(0.3)
SAGEConv(128 -> 2)
```

### 10.4 Model contract bắt buộc

Mọi model phải pass:

- `forward(data)` trả về logits `[num_nodes, num_classes]`.
- `get_embeddings(data)` trả về `[num_nodes, hidden_dim]`.
- `loss_compute(logits, y, mask)` trả về scalar tensor.
- `get_config()` có đủ `name`, `num_features`, `num_classes`, `hidden_dim`, `dropout`.

---

## 11. Loss Function & Optimization

### 11.1 Weighted Cross Entropy

Phương án an toàn, dễ implement:

```python
class_weight = compute_class_weight_from_labeled_pool(y[L])
loss = F.cross_entropy(logits[L], y[L], weight=class_weight)
```

### 11.2 Focal Loss

Dùng khi Weighted CE chưa đủ tốt.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
```

Khuyến nghị:

| Config | Giá trị |
|---|---:|
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.3 |
| Max epochs | 200 |
| Early stopping patience | 20 |
| Monitor | validation Average Precision hoặc F1-macro |

---

## 12. Active Learning Strategies

### 12.1 Random Sampling

Baseline bắt buộc.

```python
@register_strategy("random")
class RandomSampling(BaseALStrategy):
    def score_samples(self, model, graph, candidate_indices, device):
        return torch.rand(len(candidate_indices), device=device)
```

### 12.2 Entropy Sampling

Chọn node model không chắc chắn nhất.

```python
probs = torch.softmax(logits[candidate_indices], dim=-1)
scores = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
```

### 12.3 Margin Sampling

Chọn node có khoảng cách giữa top-1 và top-2 probability nhỏ nhất.

```python
probs = torch.softmax(logits[candidate_indices], dim=-1)
top2 = probs.topk(k=2, dim=-1).values
margin = top2[:, 0] - top2[:, 1]
scores = -margin
```

### 12.4 Coreset Sampling

Chọn node đa dạng trong embedding space.

```python
emb = model.get_embeddings(graph)
U_emb = emb[unlabeled_indices]
L_emb = emb[labeled_indices]
# greedy k-center: chọn node xa labeled set hiện tại nhất
```

### 12.5 Quy tắc chống leakage cho AL

- Strategy chỉ score node trong `U`.
- Strategy không nhìn nhãn thật của `U` khi score.
- Strategy không score validation/test nodes.
- Nếu cần class balancing, chỉ dùng model prediction hoặc uncertainty proxy, không dùng true label.
- Query history phải được lưu để tái lập.

---

## 13. Trainer

### 13.1 Trainer responsibilities

`Trainer` chỉ biết:

- model
- graph train view
- train mask hiện tại `L`
- validation graph view
- validation mask
- optimizer/scheduler/loss config

`Trainer` không biết AL strategy, không tự update pool.

### 13.2 Training loop

```python
class Trainer:
    def fit(self, model, train_graph, train_mask, val_graph, val_mask):
        best_metric = -float("inf")
        patience_counter = 0

        for epoch in range(self.cfg.max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(train_graph)
            loss = model.loss_compute(logits, train_graph.y, train_mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            train_metrics = self.metric_manager.compute_from_logits(
                logits.detach(), train_graph.y, train_mask
            )

            val_metrics = self.evaluate(model, val_graph, val_mask)

            self.event_logger.log_epoch(
                epoch=epoch,
                loss=float(loss.item()),
                grad_norm=float(grad_norm),
                lr=optimizer.param_groups[0]["lr"],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            current = val_metrics[self.cfg.monitor]
            if current > best_metric:
                best_metric = current
                patience_counter = 0
                self.checkpoint_manager.save_best(model, epoch, val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                break
```

---

## 14. AL Orchestrator

### 14.1 High-level loop

```text
Load config
Set seed
Load dataset
Preprocess safely
Build graph views
Initialize pool L0, U
Initialize model, strategy, tracker

For round r in 0..T-1:
    Train model on G_train + L
    Validate on G_val + val_mask
    Save checkpoint and logs
    Evaluate on G_test + test_mask with no gradient
    Score U with strategy
    Query b nodes
    Oracle reveals labels
    Update L and U
    Save pool state, round summary, artifacts

Aggregate all rounds
Export learning curves and final tables
```

### 14.2 Orchestrator contract

```python
class ALOrchestrator:
    def __init__(self, cfg, dataset, model, strategy, tracker, artifact_store, device):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.strategy = strategy
        self.tracker = tracker
        self.artifact_store = artifact_store
        self.device = device

    def run(self):
        self.validate_no_leakage()
        self.tracker.start_run(self.cfg)

        for round_id in range(self.cfg.al.num_rounds):
            self.run_round(round_id)

        self.tracker.finish()
```

### 14.3 Round summary phải lưu

```json
{
  "round": 2,
  "strategy": "entropy",
  "num_labeled": 100,
  "num_unlabeled": 3200,
  "train_loss_best": 0.38,
  "val_f1_macro_best": 0.61,
  "val_avg_precision_best": 0.42,
  "test_f1_macro": 0.58,
  "test_auc_roc": 0.86,
  "test_avg_precision": 0.39,
  "query_budget": 20,
  "epoch_best": 73,
  "early_stopped": true,
  "wall_time_sec": 84.2,
  "gpu_memory_mb": 3120
}
```

---

## 15. Evaluation Design

### 15.1 Metrics chính

| Metric | Vai trò |
|---|---|
| F1-macro | Công bằng hơn với minority class |
| Average Precision | Rất quan trọng cho imbalanced binary classification |
| AUC-ROC | Đo ranking tổng quát |
| Recall@K | Phù hợp fraud detection: trong top K cảnh báo bắt được bao nhiêu fraud |
| AUC-LC | Diện tích dưới learning curve, so sánh AL strategy qua nhiều budget |

Không dùng accuracy làm metric chính.

### 15.2 Learning curve

Trục x:

- số node đã label
- hoặc percentage of train labels used

Trục y:

- test F1-macro
- test Average Precision
- test AUC-ROC
- Recall@K

Mỗi đường:

- Random
- Entropy
- Margin
- Coreset
- GNN-full upper bound
- MLP-full baseline

### 15.3 Passive baselines

| Baseline | Mục đích |
|---|---|
| MLP-full | Không dùng graph, train full known train labels |
| GNN-random | GNN + random sampling cùng budget |
| GNN-full | Upper bound, dùng toàn bộ known train labels |
| Label Propagation | Semi-supervised baseline nếu còn thời gian |

### 15.4 Statistical testing

Giai đoạn final:

- Chạy 5 seeds tối thiểu.
- Nếu có thời gian, 10 seeds.
- Report mean ± std.
- Dùng Wilcoxon signed-rank test giữa Coreset/Entropy/Margin và Random trên AUC-LC hoặc final AP.

---

## 16. Observability: quan sát training và toàn pipeline

Đây là phần cần mở rộng nhất để project không chỉ chạy được, mà còn **kiểm tra được**.

### 16.1 Các tầng cần quan sát

| Tầng | Cần log gì? | Mục đích |
|---|---|---|
| Config | resolved config, CLI overrides | Biết chính xác run đã chạy gì |
| Environment | Python, PyTorch, PyG, CUDA, GPU | Reproduce và debug device issue |
| Data | node/edge counts, split counts, class ratios | Phát hiện split sai, imbalance sai |
| Leakage checks | temporal check, unknown check, scaler check | Fail fast trước khi train |
| Training | loss, lr, grad_norm, epoch time | Theo dõi quá trình học |
| Validation | F1, AP, AUC, early stopping | Chọn checkpoint |
| Test | F1, AP, AUC, Recall@K | Kết quả báo cáo |
| AL query | selected indices, scores, query class ratio | Hiểu strategy đang chọn gì |
| Pool | L/U sizes per round | Kiểm tra AL loop |
| Checkpoint | best epoch, best metric, path | Resume và audit |
| Artifacts | plots, CSV, JSON, model, pool state | Làm báo cáo và tái lập |

### 16.2 Tracking backend

Có thể dùng một trong ba chế độ:

| Mode | Khi dùng |
|---|---|
| Local CSV/JSON | Luôn bật, không phụ thuộc internet |
| TensorBoard | Dễ theo dõi train curves |
| W&B hoặc MLflow | Tốt cho sweep, artifact, so sánh nhiều runs |

Khuyến nghị: **Local CSV/JSON luôn bật**, W&B optional.

### 16.3 Log schema đề xuất

```text
outputs/{experiment_name}/{run_id}/
├── config/
│   ├── resolved_config.yaml
│   ├── command.txt
│   └── environment.json
├── data/
│   ├── dataset_summary.json
│   ├── split_summary.csv
│   ├── class_ratio.csv
│   └── leakage_report.json
├── rounds/
│   ├── round_000/
│   │   ├── train_log.csv
│   │   ├── val_log.csv
│   │   ├── test_metrics.json
│   │   ├── query_scores.csv
│   │   ├── queried_nodes.csv
│   │   ├── pool_state.json
│   │   └── checkpoint.pt
│   └── round_001/
├── plots/
│   ├── learning_curve_f1_macro.png
│   ├── learning_curve_avg_precision.png
│   ├── query_score_distribution_round_000.png
│   ├── class_ratio_by_round.png
│   └── embedding_tsne_queries.png
├── tables/
│   ├── final_metrics.csv
│   ├── auc_lc.csv
│   └── statistical_tests.csv
└── artifacts_manifest.json
```

### 16.4 EventLogger API

```python
class EventLogger:
    def log_config(self, cfg): ...
    def log_environment(self): ...
    def log_dataset_summary(self, summary): ...
    def log_leakage_report(self, report): ...
    def log_epoch(self, round_id, epoch, metrics): ...
    def log_round(self, round_id, summary): ...
    def log_query(self, round_id, query_df): ...
    def log_pool_state(self, round_id, pool_state): ...
    def log_artifact(self, name, path, artifact_type): ...
```

### 16.5 Metrics cần log mỗi epoch

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| loss | Có | Có | Không bắt buộc |
| F1-macro | Có | Có | Mỗi round |
| Average Precision | Có | Có | Mỗi round |
| AUC-ROC | Có | Có | Mỗi round |
| Recall@K | Không bắt buộc | Có | Có |
| LR | Có | - | - |
| Grad norm | Có | - | - |
| Epoch time | Có | - | - |
| GPU memory | Có | - | - |

### 16.6 Query observability

Mỗi round lưu `query_scores.csv`:

| node_id | time_step | score | selected | predicted_class | p_licit | p_illicit | true_label_hidden_for_audit |
|---:|---:|---:|---:|---:|---:|---:|---|

Ghi chú:

- `true_label_hidden_for_audit` chỉ được ghi sau khi query đã chọn xong, dùng để phân tích offline.
- Strategy không được truy cập cột này khi score.

### 16.7 Leakage report

Trước khi train, pipeline tạo `leakage_report.json`:

```json
{
  "temporal_split_valid": true,
  "no_test_nodes_in_train_graph": true,
  "unknown_excluded_from_loss": true,
  "unknown_excluded_from_query_pool": true,
  "scaler_fit_scope": "train_time_only",
  "test_used_for_tuning": false,
  "status": "PASS"
}
```

Nếu bất kỳ check nào fail, pipeline dừng ngay.

---

## 17. Checkpointing & Resume

### 17.1 Checkpoint nội dung

```python
checkpoint = {
    "round": round_id,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_metric": best_metric,
    "model_config": model.get_config(),
    "pool_state": pool_manager.state_dict(),
    "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    "rng_state": get_rng_state(),
}
```

### 17.2 Resume modes

| Mode | Ý nghĩa |
|---|---|
| `resume_from_checkpoint` | Resume train trong cùng round |
| `resume_from_round` | Resume từ round đã hoàn thành |
| `eval_only` | Load checkpoint và evaluate |
| `replay_queries` | Re-run pipeline với query history đã lưu |

---

## 18. Hydra Config đề xuất

### 18.1 `configs/config.yaml`

```yaml
defaults:
  - dataset: elliptic
  - model: gcn
  - al_strategy: entropy
  - trainer: default
  - tracking: local
  - experiment: debug
  - _self_

project:
  name: fraud-gnn-al

experiment:
  name: elliptic_gcn_entropy
  seed: 42
  output_dir: outputs/${experiment.name}

al:
  num_rounds: 10
  seed_size: 50
  query_budget: 20
  retrain_from_scratch_each_round: true

runtime:
  device: auto
  num_workers: 0
  deterministic: true
```

### 18.2 `configs/dataset/elliptic.yaml`

```yaml
name: elliptic
root: data/elliptic
num_classes: 2
num_features: 166
unknown_label: -1
split:
  train_end_step: 34
  val_start_step: 35
  val_end_step: 40
  test_start_step: 41
normalization:
  type: standard_scaler
  fit_scope: train_time_only
graph_views:
  train: train_time_only
  val: train_plus_val_time
  test: all_time_no_grad
```

### 18.3 `configs/model/gcn.yaml`

```yaml
name: gcn
num_features: ${dataset.num_features}
num_classes: ${dataset.num_classes}
hidden_dim: 128
num_layers: 2
dropout: 0.3
loss:
  name: focal
  gamma: 2.0
  alpha: auto_from_labeled_pool
```

### 18.4 `configs/al_strategy/coreset.yaml`

```yaml
name: coreset
embedding_source: penultimate
selection: greedy_k_center
distance: euclidean
batch_size: ${al.query_budget}
```

### 18.5 `configs/trainer/default.yaml`

```yaml
max_epochs: 200
patience: 20
monitor: val_avg_precision
monitor_mode: max
optimizer:
  name: adamw
  lr: 1e-3
  weight_decay: 1e-4
scheduler:
  name: cosine
  enabled: false
grad_clip: 5.0
```

### 18.6 `configs/tracking/local.yaml`

```yaml
backend: local
save_csv: true
save_json: true
save_plots: true
log_every_n_epochs: 1
```

---

## 19. Testing Strategy

### 19.1 Unit tests

| Test | Mục tiêu |
|---|---|
| `test_focal_loss_scalar` | Loss trả scalar |
| `test_metric_manager_binary_imbalance` | AP/F1/AUC tính đúng |
| `test_pool_update_moves_indices` | Query xong U giảm, L tăng |
| `test_entropy_scores_shape` | Strategy trả score đúng shape |
| `test_coreset_no_duplicate` | Không query trùng node |
| `test_recall_at_k` | Recall@K đúng |

### 19.2 Contract tests

Dataset contract:

- Có `data.x`, `data.edge_index`, `data.y`, `data.time_step`.
- `num_features == 166`.
- Có đủ masks: train, val, test, known, unknown.
- Unknown không nằm trong supervised train/val/test mask.
- Train/val/test time steps không overlap.

Model contract:

- `forward(graph)` trả logits `[num_nodes, 2]`.
- `get_embeddings(graph)` trả `[num_nodes, hidden_dim]`.
- `loss_compute(logits, y, mask)` trả scalar.
- Model không softmax trong forward.

Strategy contract:

- Chỉ nhận candidate_indices từ U.
- Không query quá budget.
- Không query duplicate.
- Không query node ngoài U.

### 19.3 Leakage tests

```python
def test_no_random_split_used(cfg):
    assert cfg.dataset.split.train_end_step == 34
    assert cfg.dataset.split.test_start_step == 41


def test_unknown_excluded_from_query_pool(pool_manager, data):
    assert not torch.isin(pool_manager.unlabeled_indices, unknown_indices).any()


def test_test_nodes_not_in_train_graph(graph_views, masks):
    train_graph = graph_views["train"]
    assert train_graph.time_step.max() <= 34


def test_scaler_fit_train_only(dataset):
    assert dataset.scaler_fit_scope == "train_time_only"
```

### 19.4 Integration tests

| Test | Mục tiêu |
|---|---|
| `test_single_epoch_train_gcn` | GCN train được 1 epoch |
| `test_single_al_round_entropy` | Một round AL chạy end-to-end |
| `test_checkpoint_resume` | Resume được từ checkpoint |
| `test_outputs_created` | Log, metrics, pool_state, checkpoint đều có |
| `test_reproducibility_same_seed` | Cùng seed ra cùng L0 và query random |

---

## 20. Experiment Plan

### 20.1 Debug run

```bash
python scripts/run_experiment.py experiment=debug model=gcn al_strategy=random al.num_rounds=2 trainer.max_epochs=5
```

Mục tiêu:

- Validate pipeline chạy end-to-end.
- Kiểm tra artifacts được tạo.
- Kiểm tra không leakage.

### 20.2 Main experiments

| Model | Strategy | Seeds | Budget |
|---|---|---:|---:|
| GCN | Random | 5 | 20/round |
| GCN | Entropy | 5 | 20/round |
| GCN | Margin | 5 | 20/round |
| GCN | Coreset | 5 | 20/round |
| GraphSAGE | Random | 5 | 20/round |
| GraphSAGE | Entropy | 5 | 20/round |
| GraphSAGE | Coreset | 5 | 20/round |

### 20.3 Ablation

| Ablation | Mục tiêu |
|---|---|
| query_budget: 10 / 20 / 50 | Kiểm tra budget sensitivity |
| seed_size: 20 / 50 | Kiểm tra L0 sensitivity |
| loss: Weighted CE vs Focal Loss | Kiểm tra xử lý imbalance |
| model: GCN vs GraphSAGE | Kiểm tra backbone |

---

## 21. Analysis & Visualization

### 21.1 Biểu đồ bắt buộc

| Plot | Mục đích |
|---|---|
| Node count by time step | Thấy temporal structure |
| Label ratio by time step | Thấy imbalance và distribution shift |
| Degree distribution | Thấy graph structure |
| Known labels by split | Chứng minh split hợp lý |
| Learning curve F1-macro | So sánh AL strategies |
| Learning curve Average Precision | Metric chính cho fraud |
| Query class ratio by round | AL có tìm được fraud không |
| Query score distribution | Kiểm tra strategy behavior |
| t-SNE/UMAP embeddings with queried nodes | Phân tích Coreset/diversity |

### 21.2 Error analysis

Cần trả lời:

- Fraud node nào model hay miss?
- Missed fraud tập trung ở time step nào?
- AL strategy có query nhiều fraud hơn Random không?
- Entropy có bị majority class chi phối không?
- Coreset có query đa dạng hơn không?
- GraphSAGE có ổn định hơn GCN không?

---

## 22. README structure

README nên có:

```text
1. Project title
2. Research question
3. Dataset
4. Leakage-safe data pipeline
5. Methods
   - MLP
   - GCN
   - GraphSAGE
   - Active Learning strategies
6. Metrics
7. How to install
8. How to run debug experiment
9. How to run full benchmark
10. How to reproduce results
11. Results table
12. Learning curves
13. Error analysis
14. Limitations
15. Future work
```

---

## 23. Báo cáo PDF / report 5 chương

1. **Mở đầu:** bài toán, annotation cost, class imbalance, mục tiêu.
2. **Cơ sở lý thuyết:** GNN, Active Learning, Focal Loss, Average Precision.
3. **Dữ liệu & phương pháp:** Elliptic, temporal split, unknown handling, graph views, AL loop.
4. **Thực nghiệm & đánh giá:** baselines, learning curves, ablation, statistical tests, resource report.
5. **Kết luận:** đóng góp, hạn chế, hướng phát triển.

---

## 24. Timeline 6 tuần

| Tuần | Việc chính | Output |
|---:|---|---|
| 1 | Setup, load dataset, EDA, split, scaler, leakage checks | Data pipeline hoàn chỉnh |
| 2 | MLP, GCN, GraphSAGE baseline | Baseline chạy được |
| 3 | Active Learning loop: Random, Entropy, Margin | Learning curve bản đầu |
| 4 | Coreset, Focal Loss, checkpoint, tracking | Full AL experiment |
| 5 | Ablation, statistical test, error analysis | Bảng kết quả chính |
| 6 | Viết báo cáo, slide, clean code, README | Sản phẩm nộp |

---

## 25. Acceptance Criteria

Project được xem là hoàn chỉnh khi:

- `python scripts/validate_pipeline.py` pass toàn bộ leakage checks.
- Debug run chạy end-to-end ít nhất 2 AL rounds.
- Có ít nhất MLP, GCN, Random, Entropy, Coreset.
- Mỗi round lưu được train log, val log, test metrics, query scores, pool state, checkpoint.
- Có learning curves cho F1-macro và Average Precision.
- Có bảng mean ± std qua nhiều seeds.
- Có phân tích vì sao AL tốt hơn hoặc không tốt hơn Random.
- README hướng dẫn reproduce rõ ràng.

---

## 26. Một câu mô tả project dùng cho portfolio

> This project builds a leakage-safe, observable, and reproducible Active Learning pipeline for fraud detection on the Elliptic Bitcoin transaction graph. It compares compact GNN backbones against non-graph baselines under limited annotation budgets, logs every training and query step, and evaluates strategies using imbalance-aware metrics such as Average Precision, F1-macro, AUC-ROC, Recall@K, and AUC of the learning curve.
