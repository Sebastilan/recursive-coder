"""可复用的树状可视化模块。

用法:
    from tree_viz import save_tree_html, node_to_dict

    # 方式 1：从 Node 对象转换
    data = node_to_dict(root_node)
    save_tree_html(data, "output.html", title="CVRP 分解树")

    # 方式 2：直接传 dict
    tree = {"label": "任务", "status": "no", "children": [...]}
    save_tree_html(tree, "output.html")

树节点 dict 格式:
    {
        "label": str,          # 任务描述
        "status": str,         # "yes" | "no" | "no_depth"
        "detail": str,         # Judge 完整回复（可选）
        "time": float,         # 总耗时秒（可选）
        "judge_time": float,   # Judge 耗时（可选）
        "split_time": float,   # 分解耗时（可选）
        "children": [...]      # 子节点列表（可选）
    }
"""

import json
from typing import Any


def node_to_dict(node: Any) -> dict:
    """将 Node 对象转为可视化 dict。

    兼容 test_split.py 中的 Node 类（或任何具有
    task/feasible/children/judge_response/judge_time/split_time 属性的对象）。
    """
    if node.feasible:
        status = "yes"
    elif not node.children:
        status = "no_depth"
    else:
        status = "no"

    d = {
        "label": node.task,
        "status": status,
        "detail": getattr(node, "judge_response", ""),
        "time": round(getattr(node, "judge_time", 0)
                       + getattr(node, "split_time", 0), 1),
        "judge_time": round(getattr(node, "judge_time", 0), 1),
        "split_time": round(getattr(node, "split_time", 0), 1),
    }
    if node.children:
        d["children"] = [node_to_dict(c) for c in node.children]
    return d


def _count_stats(node: dict) -> dict:
    children = node.get("children", [])
    stats = {"total": 1, "yes": 0, "no": 0, "time": node.get("time", 0)}
    if not children:
        if node.get("status") == "yes":
            stats["yes"] = 1
        else:
            stats["no"] = 1
    for child in children:
        cs = _count_stats(child)
        for k in ("total", "yes", "no", "time"):
            stats[k] += cs[k]
    return stats


def save_tree_html(
    tree_data: dict,
    output_path: str,
    title: str = "二分分解树",
) -> str:
    """生成交互式树可视化 HTML 并保存。返回输出文件路径。"""
    stats = _count_stats(tree_data)
    tree_json = json.dumps(tree_data, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)

    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__TREE_JSON__", tree_json)
            .replace("__STATS_JSON__", stats_json))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── HTML 模板 ─────────────────────────────────────────────────────────────────

_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: "Microsoft YaHei", "PingFang SC", -apple-system, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  overflow: hidden;
  height: 100vh;
}

/* ── 顶栏 ── */
#header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: #1e293b;
  border-bottom: 1px solid #334155;
  height: 56px;
  z-index: 10;
}
#header h1 {
  font-size: 16px;
  font-weight: 600;
  color: #f1f5f9;
}
.stats {
  display: flex;
  gap: 20px;
  font-size: 13px;
  color: #94a3b8;
}
.stats .stat-item { display: flex; align-items: center; gap: 6px; }
.stats .dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block;
}
.dot-yes { background: #22c55e; }
.dot-no { background: #ef4444; }
.dot-depth { background: #f97316; }

/* ── 图例 ── */
.legend {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #64748b;
  padding: 8px 24px;
  background: #0f172a;
}
.legend-item { display: flex; align-items: center; gap: 4px; }

/* ── 树容器 ── */
#tree-container {
  width: 100vw;
  height: calc(100vh - 56px);
  cursor: grab;
}
#tree-container:active { cursor: grabbing; }
#tree-svg { width: 100%; height: 100%; }

/* ── 连线 ── */
.link {
  fill: none;
  stroke: #334155;
  stroke-width: 1.5;
}

/* ── 节点框 ── */
.node-box {
  width: 220px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 10px 12px;
  cursor: pointer;
  transition: box-shadow 0.2s, border-color 0.2s;
  overflow: hidden;
}
.node-box:hover {
  box-shadow: 0 0 16px rgba(99, 102, 241, 0.3);
  border-color: #6366f1;
}
.node-box.active {
  border-color: #818cf8;
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}

/* 状态条 */
.node-box.status-yes    { border-left: 4px solid #22c55e; }
.node-box.status-no     { border-left: 4px solid #ef4444; }
.node-box.status-no_depth { border-left: 4px solid #f97316; }

.node-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.badge {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 4px;
  font-weight: 700;
  color: #fff;
}
.badge-yes      { background: #22c55e; }
.badge-no       { background: #ef4444; }
.badge-no_depth { background: #f97316; }

.node-time {
  font-size: 11px;
  color: #64748b;
}
.node-label {
  font-size: 12px;
  color: #cbd5e1;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ── 详情面板 ── */
#detail {
  position: fixed;
  top: 56px;
  right: 0;
  width: 360px;
  height: calc(100vh - 56px);
  background: #1e293b;
  border-left: 1px solid #334155;
  padding: 20px;
  overflow-y: auto;
  transform: translateX(100%);
  transition: transform 0.25s ease;
  z-index: 20;
}
#detail.open { transform: translateX(0); }

#detail .close-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  background: none;
  border: none;
  color: #64748b;
  font-size: 20px;
  cursor: pointer;
}
#detail .close-btn:hover { color: #e2e8f0; }

#detail h3 {
  font-size: 14px;
  color: #94a3b8;
  margin: 16px 0 6px 0;
  font-weight: 500;
}
#detail h3:first-of-type { margin-top: 0; }

#detail .detail-text {
  font-size: 13px;
  color: #e2e8f0;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
#detail .detail-badge {
  display: inline-block;
  margin-bottom: 12px;
}

/* ── 子任务可点击 ── */
.child-item {
  margin: 6px 0;
  padding: 8px 10px;
  background: #0f172a;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: border-color 0.2s, background 0.2s;
  line-height: 1.5;
}
.child-item:hover {
  border-color: #6366f1;
  background: #162032;
}
.child-item .child-label {
  margin-top: 4px;
  color: #cbd5e1;
}

/* ── 提示 ── */
.hint {
  position: fixed;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 12px;
  color: #475569;
  pointer-events: none;
}
</style>
</head>
<body>

<div id="header">
  <h1>__TITLE__</h1>
  <div class="stats" id="stats-bar"></div>
</div>

<div id="tree-container">
  <svg id="tree-svg"></svg>
</div>

<div id="detail">
  <button class="close-btn" onclick="closeDetail()">&times;</button>
  <div id="detail-content"></div>
</div>

<div class="hint">滚轮缩放 &middot; 拖拽平移 &middot; 点击节点查看详情</div>

<script>
// ── 数据 ──
const treeData = __TREE_JSON__;
const stats = __STATS_JSON__;

const STATUS = {
  yes:      { text: "可行",     color: "#22c55e", icon: "\u2714" },
  no:       { text: "不可行",   color: "#ef4444", icon: "\u2716" },
  no_depth: { text: "深度限制", color: "#f97316", icon: "\u26A0" },
};

// ── 统计栏 ──
document.getElementById("stats-bar").innerHTML = [
  '<span class="stat-item"><span class="dot dot-yes"></span>可行叶子: ' + stats.yes + '</span>',
  '<span class="stat-item"><span class="dot dot-no"></span>不可行叶子: ' + stats.no + '</span>',
  '<span class="stat-item">总节点: ' + stats.total + '</span>',
  '<span class="stat-item">总耗时: ' + stats.time.toFixed(1) + 's</span>',
].join("");

// ── 布局 ──
const BOX_W = 220, BOX_H = 80;
const NODE_W = BOX_W + 40;   // 节点水平间距
const NODE_H = BOX_H + 50;   // 节点垂直间距

const svg = d3.select("#tree-svg");
const containerEl = document.getElementById("tree-container");
const width = containerEl.clientWidth;
const height = containerEl.clientHeight;

svg.attr("width", width).attr("height", height);

const g = svg.append("g");

// 缩放 & 平移
const zoom = d3.zoom()
  .scaleExtent([0.1, 3])
  .on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

// ── D3 树 ──
const root = d3.hierarchy(treeData, d => d.children);

// 给每个节点分配唯一 ID
root.descendants().forEach((d, i) => { d.data._id = i; });

const treeLayout = d3.tree().nodeSize([NODE_W, NODE_H]);
treeLayout(root);

// 节点映射表（ID → d3 节点 + DOM 元素）
const nodeRef = {};

// 初始居中
const initialX = width / 2;
const initialY = 60;
svg.call(zoom.transform, d3.zoomIdentity.translate(initialX, initialY));

// ── 连线 ──
g.selectAll(".link")
  .data(root.links())
  .join("path")
  .attr("class", "link")
  .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

// ── 节点 ──
const nodes = g.selectAll(".node")
  .data(root.descendants())
  .join("g")
  .attr("class", "node")
  .attr("transform", d => "translate(" + d.x + "," + d.y + ")");

nodes.each(function(d) {
  const fo = d3.select(this).append("foreignObject")
    .attr("width", BOX_W)
    .attr("height", BOX_H)
    .attr("x", -BOX_W / 2)
    .attr("y", -BOX_H / 2);

  const s = d.data.status || "no";
  const info = STATUS[s] || STATUS.no;
  const label = d.data.label || "";
  const truncated = label.length > 45 ? label.slice(0, 45) + "\u2026" : label;
  const timeStr = (d.data.time || 0).toFixed(1) + "s";

  const div = fo.append("xhtml:div")
    .attr("class", "node-box status-" + s)
    .attr("data-id", d.data._id || "");

  div.html(
    '<div class="node-header">' +
      '<span class="badge badge-' + s + '">' + info.text + '</span>' +
      '<span class="node-time">' + timeStr + '</span>' +
    '</div>' +
    '<div class="node-label">' + escapeHtml(truncated) + '</div>'
  );

  div.on("click", function(event) {
    event.stopPropagation();
    d3.selectAll(".node-box").classed("active", false);
    d3.select(this).classed("active", true);
    showDetail(d.data);
  });

  // 存储引用
  nodeRef[d.data._id] = { d3Node: d, dom: div.node() };
});

// ── 详情面板 ──
function showDetail(data) {
  const s = data.status || "no";
  const info = STATUS[s] || STATUS.no;
  const panel = document.getElementById("detail-content");

  let html = '<span class="badge badge-' + s + ' detail-badge">' + info.text + '</span>';

  html += '<h3>任务描述</h3>';
  html += '<div class="detail-text">' + escapeHtml(data.label || "") + '</div>';

  if (data.detail) {
    html += '<h3>Judge 回复</h3>';
    html += '<div class="detail-text">' + escapeHtml(data.detail) + '</div>';
  }

  html += '<h3>耗时</h3>';
  html += '<div class="detail-text">';
  if (data.judge_time !== undefined) {
    html += '判断: ' + data.judge_time + 's';
  }
  if (data.split_time) {
    html += '  |  分解: ' + data.split_time + 's';
  }
  html += '  |  合计: ' + (data.time || 0).toFixed(1) + 's';
  html += '</div>';

  if (data.children && data.children.length) {
    html += '<h3>子任务 (' + data.children.length + ')  <span style="font-size:11px;color:#64748b;font-weight:400">点击跳转</span></h3>';
    data.children.forEach(function(c, i) {
      const cs = STATUS[c.status] || STATUS.no;
      html += '<div class="child-item" onclick="navigateToNode(' + c._id + ')">';
      html += '<span class="badge badge-' + c.status + '" style="margin-right:6px">' + cs.text + '</span>';
      html += '<span class="node-time" style="margin-left:4px">' + (c.time || 0).toFixed(1) + 's</span>';
      html += '<div class="child-label">' + escapeHtml(c.label) + '</div>';
      html += '</div>';
    });
  }

  panel.innerHTML = html;
  document.getElementById("detail").classList.add("open");
}

function closeDetail() {
  document.getElementById("detail").classList.remove("open");
  d3.selectAll(".node-box").classed("active", false);
}

function navigateToNode(id) {
  const ref = nodeRef[id];
  if (!ref) return;

  // 高亮目标节点
  d3.selectAll(".node-box").classed("active", false);
  d3.select(ref.dom).classed("active", true);

  // 平滑移动到目标节点（居中显示）
  const d = ref.d3Node;
  svg.transition().duration(500).call(
    zoom.transform,
    d3.zoomIdentity.translate(width / 2 - d.x, height / 2 - d.y)
  );

  // 显示目标节点详情
  showDetail(d.data);
}

// 点击空白关闭面板
svg.on("click", closeDetail);

// ── 工具函数 ──
function escapeHtml(text) {
  var div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>
'''


# ── 命令行测试 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example = {
        "label": "实现一个完整的 C++ CVRP Branch-and-Price 算法",
        "status": "no",
        "detail": "NO - 需要多个复杂模块协作",
        "time": 2.3, "judge_time": 1.5, "split_time": 0.8,
        "children": [
            {
                "label": "实现主问题框架（列生成循环 + RMP）",
                "status": "no",
                "detail": "NO - 仍需要 LP 求解器 + 列管理",
                "time": 1.8, "judge_time": 1.2, "split_time": 0.6,
                "children": [
                    {"label": "实现受限主问题 RMP 的 LP 建模",
                     "status": "yes", "detail": "YES - 单个 LP 建模任务",
                     "time": 0.9, "judge_time": 0.9, "split_time": 0},
                    {"label": "实现列生成迭代循环与收敛判断",
                     "status": "yes", "detail": "YES - 标准迭代逻辑",
                     "time": 0.7, "judge_time": 0.7, "split_time": 0},
                ],
            },
            {
                "label": "实现 SPPRC 定价子问题求解器",
                "status": "no_depth",
                "detail": "NO - 标签设定算法 + 支配规则较复杂",
                "time": 1.1, "judge_time": 1.1, "split_time": 0,
            },
        ],
    }
    save_tree_html(example, "experiments/example_tree.html", title="示例：CVRP 分解树")
    print("已保存: experiments/example_tree.html")
