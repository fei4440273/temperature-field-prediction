"""Only after ROOT evidence registration, publish the signed F3 main curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from sic_cu.config import PROJECT_ROOT
from sic_cu.eval.task09_f3_main_aggregate import (
    TASK09_ROOT, aggregate_task09_main_from_signed_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="任09 F3主序列十五身份正式聚合")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if (destination.parent != TASK09_ROOT.resolve()
            or not re.fullmatch(
                r"正式F3主序列汇总_录\d{4}_\d{8}T\d{6}\+0800", destination.name)
            or destination.exists()):
        raise ValueError("任09正式汇总必须是任务09目录内唯一全新带账行目录，不可覆盖")
    result = aggregate_task09_main_from_signed_catalog(
        registry_path=args.registry, registry_sha=args.registry_sha256,
        archive_path=args.source_archive, archive_sha=args.source_archive_sha256,
        evidence_path=args.evidence, evidence_sha=args.evidence_sha256,
    )
    destination.mkdir(exist_ok=False)
    result.update({
        "正式聚合事前登记SHA256": args.registry_sha256,
        "正式聚合源码冻结tar_SHA256": args.source_archive_sha256,
        "十五身份全原件证据清单SHA256": args.evidence_sha256,
        "旧固定TEST温度读取": False,
    })
    (destination / "F3主序列十五身份机器摘要.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (destination / "F3主序列HF数量原点.csv").open(
        "x", newline="", encoding="utf-8-sig",
    ) as stream:
        fields = (
            "HF功率数", "观测最佳S_均值_摄氏度", "观测最佳S_样本标准差_摄氏度",
            "训练末S_均值_摄氏度", "训练末S_样本标准差_摄氏度",
            "best能源安全种子数", "final能源安全种子数",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: point[field] for field in fields}
                         for point in result["主序列三个原点"])
    with (destination / "F3主序列五seed逐模型.csv").open(
        "x", newline="", encoding="utf-8-sig",
    ) as stream:
        fields = (
            "HF功率数", "seed", "来源", "canonical目录",
            "best合法S_摄氏度", "final合法S_摄氏度",
            "best原能源宏均值_瓦", "final原能源宏均值_瓦",
            "best吸收功率归一宏均值", "final吸收功率归一宏均值",
            "best能源资格", "final能源资格",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item[field] for field in fields}
                         for item in result["十五seed逐模型摘要"])
    lines = [
        "# 任09 F3主序列五种子正式聚合与受限结论", "",
        "十五模型身份、训练原件和各best/final双阶独立能源均按事前登记和后验封存清单逐SHA通过；"
        "这里仅报告合法开发验证观测和名义工程能源，不包含旧固定TEST温度。", "",
        "| 新主序列HF训练数 | 观测最佳S均值±样本标准差/℃ | 训练末S均值±样本标准差/℃ | best/final能源合格种子数 |",
        "|---:|---:|---:|---:|",
    ]
    for point in result["主序列三个原点"]:
        lines.append(
            f'| {point["HF功率数"]} | '
            f'{point["观测最佳S_均值_摄氏度"]:.6f} ± {point["观测最佳S_样本标准差_摄氏度"]:.6f} | '
            f'{point["训练末S_均值_摄氏度"]:.6f} ± {point["训练末S_样本标准差_摄氏度"]:.6f} | '
            f'{point["best能源安全种子数"]}/5，{point["final能源安全种子数"]}/5 |'
        )
    reference = result["HF12历史参照_不属新三档同轨迹"]
    lines.extend([
        "", f'任07乙旧12 HF历史观测合法S：{reference["合法S_均值_摄氏度"]:.6f}'
        f' ± {reference["合法S_样本标准差_摄氏度"]:.6f}℃（样本ddof=1）；'
        "这是独立历史参照，不能与新3/6/9档合并成同一优化轨迹或作严格HF数量因果声明。",
        "", "三个主原点不论是否非单调均保留；五seed波动仅是种子变异而非子集变异。"
        "事前没有固定目标误差或所需HF数量判定统计，required HF count和节省比例均为null。"
        "名义能源积分恒等式自洽不等于原装置工程能源安全。",
        "", f'登记SHA：{args.registry_sha256}；源码tar SHA：{args.source_archive_sha256}；'
        f'十五身份全原件清单SHA：{args.evidence_sha256}。', "",
    ])
    (destination / "F3主序列正式中文验收报告.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    print(json.dumps({"正式汇总目录": str(destination),
                      "原点数": 3, "模型数": 15,
                      "required_HF_count": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
