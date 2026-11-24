# Numerai 预测研究 v1.0

用同一组按时间划分的训练和验证样本对比十种模型方案。包括 XGBoost、两周历史特征、线性模型、树模型和 MLP。看板展示逐期 Rank IC、模型对比和逐标的预测排序。

## 查看演示

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory dashboard
```

打开 [本地研究看板](http://127.0.0.1:8000)。仓库内的看板样例来自合成数据。

## 重跑合成样例

推荐 Python 3.13。安装 `requirements.txt` 后，以独立工作目录保存数据和结果。

```sh
export NUMERAI_WORK_DIR="$PWD/work/synthetic"
python generate_demo_data.py
python run_demo.py
python compare_sequence.py
python run_ten_models.py
python build_dashboard.py
python -m http.server 8000 --bind 127.0.0.1 --directory "$NUMERAI_WORK_DIR/dashboard"
```

## 使用 Numerai 数据

将 `NUMERAI_WORK_DIR` 指向另一空目录。从 `python run_demo.py` 开始，会通过 Numerai 公共接口读取所需的 v6.0 数据列。后续脚本顺序相同。脚本只生成本地预测文件，Rank IC 用作排序诊断。

可选的模型辅助搜索通过 `DEEPSEEK_API_KEY` 环境变量读取自备密钥，使用 `DEEPSEEK_MODEL` 指定模型 ID。完成十方案对比后，运行 `python agent_search.py` 生成并验证有界模型配方。真实数据的后续留出集可用 `python prepare_agent_test.py` 准备，随后运行 `python evaluate_agent_holdout.py`。候选方案在留出集打分之前冻结。
