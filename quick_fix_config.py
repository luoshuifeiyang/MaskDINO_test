#!/usr/bin/env python3
"""
快速配置修复脚本 - 专门处理 STEM_TYPE 问题
"""

import yaml
import os
import shutil
from pathlib import Path


def quick_fix_config(input_file, output_file=None):
    """
    快速修复配置文件中的已知问题
    """
    if output_file is None:
        output_file = input_file.replace('.yaml', '_fixed.yaml')

    print(f"�� 修复配置文件: {input_file}")
    print(f"�� 输出文件: {output_file}")

    # 读取配置文件
    try:
        with open(input_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 无法读取配置文件: {e}")
        return False

    # 记录修改
    changes = []

    # 1. 移除 MODEL.RESNETS.STEM_TYPE
    if ('MODEL' in config and
            'RESNETS' in config['MODEL'] and
            'STEM_TYPE' in config['MODEL']['RESNETS']):
        old_value = config['MODEL']['RESNETS']['STEM_TYPE']
        del config['MODEL']['RESNETS']['STEM_TYPE']
        changes.append(f"移除 MODEL.RESNETS.STEM_TYPE: {old_value}")

    # 2. 移除 MODEL.RESNETS.STEM_OUT_CHANNELS
    if ('MODEL' in config and
            'RESNETS' in config['MODEL'] and
            'STEM_OUT_CHANNELS' in config['MODEL']['RESNETS']):
        old_value = config['MODEL']['RESNETS']['STEM_OUT_CHANNELS']
        del config['MODEL']['RESNETS']['STEM_OUT_CHANNELS']
        changes.append(f"移除 MODEL.RESNETS.STEM_OUT_CHANNELS: {old_value}")

    # 3. 移除 MODEL.BACKBONE.FREEZE_AT（如果存在问题）
    if ('MODEL' in config and
            'BACKBONE' in config['MODEL'] and
            'FREEZE_AT' in config['MODEL']['BACKBONE']):

        # 只有当它的值可能有问题时才移除
        freeze_at = config['MODEL']['BACKBONE']['FREEZE_AT']
        if freeze_at < 0 or freeze_at > 5:  # 通常的合理范围
            del config['MODEL']['BACKBONE']['FREEZE_AT']
            changes.append(f"移除有问题的 MODEL.BACKBONE.FREEZE_AT: {freeze_at}")

    # 4. 确保有基本的评估配置
    if 'MODEL' not in config:
        config['MODEL'] = {}
        changes.append("添加 MODEL 配置")

    if 'ROI_HEADS' not in config['MODEL']:
        config['MODEL']['ROI_HEADS'] = {}
        changes.append("添加 MODEL.ROI_HEADS 配置")

    if 'SCORE_THRESH_TEST' not in config['MODEL']['ROI_HEADS']:
        config['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = 0.5
        changes.append("设置默认置信度阈值: 0.5")

    # 5. 保存修复后的配置
    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print("✅ 修复完成!")
        print("�� 修改内容:")
        for change in changes:
            print(f"   - {change}")

        if not changes:
            print("   - 没有发现需要修复的问题")

        return True

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def test_fixed_config(config_file):
    """
    测试修复后的配置是否可以正常加载
    """
    print(f"\n�� 测试修复后的配置...")

    try:
        import sys
        sys.path.insert(0, '.')

        from detectron2.config import get_cfg

        # 先尝试不添加MaskDINO配置
        cfg = get_cfg()

        try:
            from maskdino import add_maskdino_config
            add_maskdino_config(cfg)
            print("✅ MaskDINO配置加载成功")
        except Exception as e:
            print(f"⚠️  MaskDINO配置加载失败: {e}")
            print("   使用基础detectron2配置")

        # 尝试加载配置文件
        cfg.merge_from_file(config_file)
        print("✅ 配置文件加载成功")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="快速修复MaskDINO配置文件")
    parser.add_argument(
        "--config",
        default="configs/custom/custom_maskdino.yaml",
        help="要修复的配置文件"
    )
    parser.add_argument(
        "--output",
        help="输出文件路径（默认为 原文件名_fixed.yaml）"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试修复后的配置"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="备份原始配置文件"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return

    # 备份原文件
    if args.backup:
        backup_file = args.config + '.backup'
        shutil.copy2(args.config, backup_file)
        print(f"�� 已备份原文件到: {backup_file}")

    # 修复配置
    success = quick_fix_config(args.config, args.output)

    if success and args.test:
        output_file = args.output or args.config.replace('.yaml', '_fixed.yaml')
        test_fixed_config(output_file)

    # 提供使用建议
    if success:
        output_file = args.output or args.config.replace('.yaml', '_fixed.yaml')
        print(f"\n�� 使用修复后的配置运行可视化:")
        print(f"python minimal_visualize.py \\")
        print(f"    --config-file {output_file} \\")
        print(f"    --weights output/model_final.pth \\")
        print(f"    --output-dir ./visualization_results")


if __name__ == "__main__":
    main()