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
        "order": str,          # "serial" | "parallel"（可选）
        "produces": str,       # 产出描述（可选）
        "context": str,        # 已有数据描述（可选）
        "exec_plan": [str],    # 执行计划步骤（可选）
        "exec_blockers": [str],# 缺失的前置条件（可选）
        "exec_output": str,    # 执行产出（可选）
        "exec_time": float,    # 执行规划耗时（可选）
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
        # 新增字段
        "order": getattr(node, "order", ""),
        "produces": getattr(node, "produces", ""),
        "context": getattr(node, "context", ""),
        "exec_plan": getattr(node, "exec_plan", []),
        "exec_blockers": getattr(node, "exec_blockers", []),
        "exec_output": getattr(node, "exec_output", ""),
        "exec_time": round(getattr(node, "exec_time", 0), 1),
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
.link-serial {
  stroke: #6366f1;
  stroke-width: 2;
}
.link-parallel {
  stroke: #334155;
  stroke-width: 1.5;
  stroke-dasharray: 6,3;
}

/* 连线标签 */
.link-label {
  font-size: 10px;
  fill: #64748b;
  text-anchor: middle;
}

/* ── 节点框 ── */
.node-box {
  width: 240px;
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

.order-badge {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: uppercase;
}
.order-serial   { background: #312e81; color: #a5b4fc; }
.order-parallel { background: #1e3a5f; color: #7dd3fc; }

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
.node-produces {
  font-size: 10px;
  color: #64748b;
  margin-top: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.node-exec-info {
  font-size: 10px;
  margin-top: 4px;
  color: #94a3b8;
}
.node-exec-info .blocker-tag {
  color: #f87171;
  font-weight: 600;
}

/* ── 详情面板 ── */
#detail {
  position: fixed;
  top: 56px;
  right: 0;
  width: 400px;
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

.context-box {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 120px;
  overflow-y: auto;
}

.exec-step {
  padding: 4px 0;
  font-size: 12px;
  color: #cbd5e1;
  line-height: 1.5;
  border-bottom: 1px solid #1e293b;
}
.exec-step:last-child { border-bottom: none; }
.exec-step .step-num {
  color: #6366f1;
  font-weight: 600;
  margin-right: 6px;
}

.blocker-item {
  padding: 6px 10px;
  background: #2d1215;
  border: 1px solid #7f1d1d;
  border-radius: 6px;
  font-size: 12px;
  color: #fca5a5;
  margin: 4px 0;
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
const BOX_W = 240, BOX_H = 100;
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
  .attr("class", d => {
    const parentOrder = d.source.data.order || "";
    if (parentOrder === "serial") return "link link-serial";
    if (parentOrder === "parallel") return "link link-parallel";
    return "link";
  })
  .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

// 连线标签（A/B + SERIAL/PARALLEL）
g.selectAll(".link-label")
  .data(root.links())
  .join("text")
  .attr("class", "link-label")
  .attr("x", d => (d.source.x + d.target.x) / 2)
  .attr("y", d => (d.source.y + d.target.y) / 2 - 6)
  .text(d => {
    const parentOrder = d.source.data.order || "";
    if (!parentOrder) return "";
    const children = d.source.data.children || [];
    const idx = children.indexOf(d.target.data);
    const letter = idx === 0 ? "A" : "B";
    return letter;
  });

// ── 节点 ──
const nodes = g.selectAll(".node")
  .data(root.descendants())
  .join("g")
  .attr("class", "node")
  .attr("transform", d => "translate(" + d.x + "," + d.y + ")");

nodes.each(function(d) {
  const data = d.data;
  const s = data.status || "no";
  const info = STATUS[s] || STATUS.no;
  const label = data.label || "";
  const truncated = label.length > 50 ? label.slice(0, 50) + "\u2026" : label;
  const timeStr = (data.time || 0).toFixed(1) + "s";
  const produces = data.produces || "";
  const order = data.order || "";
  const execPlan = data.exec_plan || [];
  const execBlockers = data.exec_blockers || [];

  // 计算节点框高度
  let boxH = BOX_H;
  if (produces) boxH += 16;
  if (execPlan.length > 0) boxH += 16;

  const fo = d3.select(this).append("foreignObject")
    .attr("width", BOX_W)
    .attr("height", boxH)
    .attr("x", -BOX_W / 2)
    .attr("y", -BOX_H / 2);

  const div = fo.append("xhtml:div")
    .attr("class", "node-box status-" + s)
    .attr("data-id", data._id || "");

  // 构建 HTML
  let html = '<div class="node-header">';
  html += '<span>';
  html += '<span class="badge badge-' + s + '">' + info.text + '</span>';
  if (order) {
    html += ' <span class="order-badge order-' + order + '">' + order + '</span>';
  }
  html += '</span>';
  html += '<span class="node-time">' + timeStr + '</span>';
  html += '</div>';
  html += '<div class="node-label">' + escapeHtml(truncated) + '</div>';

  if (produces) {
    const pTrunc = produces.length > 40 ? produces.slice(0, 40) + "\u2026" : produces;
    html += '<div class="node-produces">\u2192 ' + escapeHtml(pTrunc) + '</div>';
  }

  if (s === "yes" && execPlan.length > 0) {
    html += '<div class="node-exec-info">';
    html += execPlan.length + ' steps';
    if (execBlockers.length > 0) {
      html += ' <span class="blocker-tag">' + execBlockers.length + ' blocker(s)</span>';
    }
    html += '</div>';
  }

  div.html(html);

  div.on("click", function(event) {
    event.stopPropagation();
    d3.selectAll(".node-box").classed("active", false);
    d3.select(this).classed("active", true);
    showDetail(data);
  });

  // 存储引用
  nodeRef[data._id] = { d3Node: d, dom: div.node() };
});

// ── 详情面板 ──
function showDetail(data) {
  const s = data.status || "no";
  const info = STATUS[s] || STATUS.no;
  const panel = document.getElementById("detail-content");
  const order = data.order || "";
  const produces = data.produces || "";
  const context = data.context || "";
  const execPlan = data.exec_plan || [];
  const execBlockers = data.exec_blockers || [];
  const execOutput = data.exec_output || "";
  const execTime = data.exec_time || 0;

  let html = '<span class="badge badge-' + s + ' detail-badge">' + info.text + '</span>';
  if (order) {
    html += ' <span class="order-badge order-' + order + ' detail-badge">' + order.toUpperCase() + '</span>';
  }

  html += '<h3>任务描述</h3>';
  html += '<div class="detail-text">' + escapeHtml(data.label || "") + '</div>';

  // 已有数据
  if (context) {
    html += '<h3>已有数据 (Context)</h3>';
    html += '<div class="context-box">' + escapeHtml(context) + '</div>';
  }

  // 产出
  if (produces) {
    html += '<h3>产出 (Produces)</h3>';
    html += '<div class="detail-text">' + escapeHtml(produces) + '</div>';
  }

  if (data.detail) {
    html += '<h3>Judge 回复</h3>';
    html += '<div class="detail-text">' + escapeHtml(data.detail) + '</div>';
  }

  // 执行计划
  if (execPlan.length > 0) {
    html += '<h3>执行计划 (' + execPlan.length + ' steps, ' + execTime.toFixed(1) + 's)</h3>';
    html += '<div>';
    execPlan.forEach(function(step, i) {
      html += '<div class="exec-step"><span class="step-num">' + (i+1) + '.</span>' + escapeHtml(step) + '</div>';
    });
    html += '</div>';
    if (execOutput) {
      html += '<h3>执行产出</h3>';
      html += '<div class="detail-text">' + escapeHtml(execOutput) + '</div>';
    }
  }

  // Blockers
  if (execBlockers.length > 0) {
    html += '<h3>缺失依赖 (' + execBlockers.length + ')</h3>';
    execBlockers.forEach(function(b) {
      html += '<div class="blocker-item">' + escapeHtml(b) + '</div>';
    });
  }

  html += '<h3>耗时</h3>';
  html += '<div class="detail-text">';
  if (data.judge_time !== undefined) {
    html += '判断: ' + data.judge_time + 's';
  }
  if (data.split_time) {
    html += '  |  分解: ' + data.split_time + 's';
  }
  if (execTime > 0) {
    html += '  |  规划: ' + execTime.toFixed(1) + 's';
  }
  html += '  |  合计: ' + (data.time || 0).toFixed(1) + 's';
  html += '</div>';

  if (data.children && data.children.length) {
    html += '<h3>子任务 (' + data.children.length + ')  <span style="font-size:11px;color:#64748b;font-weight:400">点击跳转</span></h3>';
    data.children.forEach(function(c, i) {
      const cs = STATUS[c.status] || STATUS.no;
      const letter = i === 0 ? "A" : "B";
      html += '<div class="child-item" onclick="navigateToNode(' + c._id + ')">';
      html += '<span style="color:#6366f1;font-weight:600;margin-right:4px">' + letter + '</span>';
      html += '<span class="badge badge-' + c.status + '" style="margin-right:6px">' + cs.text + '</span>';
      html += '<span class="node-time" style="margin-left:4px">' + (c.time || 0).toFixed(1) + 's</span>';
      html += '<div class="child-label">' + escapeHtml(c.label) + '</div>';
      if (c.produces) {
        html += '<div style="font-size:11px;color:#64748b;margin-top:2px">\u2192 ' + escapeHtml(c.produces) + '</div>';
      }
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
        "order": "serial",
        "produces": "",
        "context": "",
        "exec_plan": [], "exec_blockers": [], "exec_output": "", "exec_time": 0,
        "children": [
            {
                "label": "获取 TSPLIB 基准实例 + 研究 B&P 算法细节",
                "status": "yes",
                "detail": "YES - 数据获取 + 文献研究可在单次会话完成",
                "time": 0.9, "judge_time": 0.9, "split_time": 0,
                "order": "",
                "produces": "TSPLIB 数据文件 + B&P 算法伪代码和论文笔记",
                "context": "",
                "exec_plan": [
                    "use web_search to find TSPLIB CVRP benchmark instances",
                    "fetch_page to download .vrp files from TSPLIB mirror",
                    "use web_search for Branch-and-Price CVRP papers",
                    "summarize algorithm steps and pseudocode",
                ],
                "exec_blockers": [],
                "exec_output": "TSPLIB .vrp files + algorithm notes",
                "exec_time": 1.2,
            },
            {
                "label": "基于获取的数据实现 B&P 算法",
                "status": "no",
                "detail": "NO - 仍需要多个子模块",
                "time": 1.8, "judge_time": 1.2, "split_time": 0.6,
                "order": "parallel",
                "produces": "",
                "context": "Output from prior task: TSPLIB 数据文件 + B&P 算法伪代码和论文笔记",
                "exec_plan": [], "exec_blockers": [], "exec_output": "", "exec_time": 0,
                "children": [
                    {
                        "label": "实现列生成主问题框架（RMP + LP 求解）",
                        "status": "yes",
                        "detail": "YES - 单个 LP 建模任务",
                        "time": 0.9, "judge_time": 0.9, "split_time": 0,
                        "order": "",
                        "produces": "C++ RMP 求解模块",
                        "context": "Output from prior task: TSPLIB 数据文件 + B&P 算法伪代码和论文笔记",
                        "exec_plan": [
                            "Create C++ project structure with CMakeLists.txt",
                            "Implement RMP class with LP relaxation",
                            "Add column management (add/remove columns)",
                            "Write unit tests for RMP",
                        ],
                        "exec_blockers": [],
                        "exec_output": "rmp.cpp + rmp.h",
                        "exec_time": 1.5,
                    },
                    {
                        "label": "实现 SPPRC 定价子问题求解器",
                        "status": "yes",
                        "detail": "YES - 标签设定算法可在单次会话实现",
                        "time": 0.7, "judge_time": 0.7, "split_time": 0,
                        "order": "",
                        "produces": "C++ SPPRC 求解模块",
                        "context": "Output from prior task: TSPLIB 数据文件 + B&P 算法伪代码和论文笔记",
                        "exec_plan": [
                            "Implement label-setting algorithm for ESPPRC",
                            "Add dominance rules for label pruning",
                            "Implement resource constraints (capacity, time)",
                            "Write tests with small graph instances",
                        ],
                        "exec_blockers": [],
                        "exec_output": "spprc.cpp + spprc.h",
                        "exec_time": 1.3,
                    },
                ],
            },
        ],
    }
    save_tree_html(example, "experiments/example_tree.html", title="示例：CVRP 分解树 (v2)")
    print("已保存: experiments/example_tree.html")
