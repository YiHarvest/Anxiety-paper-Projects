"""
批量重新运行所有 step 脚本,生成使用 Cell Metabolism 配色的图表。

此脚本将:
1. 运行 step2-step8 所有脚本
2. 生成使用新 biomedical palette 的图表
3. 保持所有模型训练逻辑、数值、阈值不变

注意:运行时间较长,请耐心等待。
"""

import sys
sys.path.insert(0, '/home/yqy/Code/AnxietyProjects')

from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parent

print("=" * 80)
print("批量重新运行所有 step 脚本 (Cell Metabolism 配色)")
print("=" * 80)

print("\n配色方案:")
print("  - BIOMED_PALETTE: Deep slate blue (#4F587D), Muted mauve (#C68DC0), etc.")
print("  - GROUP_COLORS: No Anxiety (#4F587D), Anxiety (#C68DC0)")
print("  - MODEL_COLORS: Six_XGBoost (#4F587D), Six_RF (#776B97), etc.")

steps = [
    ("step2_preprocess_abis.py", "Step 2: Preprocessing & ABIS model"),
    ("step3_single_six_models.py", "Step 3: Single & Six biomarker models"),
    ("step3_gaussian_augmentation.py", "Step 3: Gaussian augmentation experiment"),
    ("step4_ratio_integrated_models.py", "Step 4: Ratio & Integrated models"),
    ("step5_abis_bootstrap_compare.py", "Step 5: Bootstrap CI & ABIS comparison"),
    ("step6_model_interpretation.py", "Step 6: Model interpretation (SHAP)"),
    ("step7_final_summary.py", "Step 7: Final summary"),
    ("step8_calibration_dca_sensitivity.py", "Step 8: Calibration, DCA & sensitivity"),
]

print(f"\n将运行 {len(steps)} 个脚本...")
print("=" * 80)

total_time = 0
successful = 0
failed = 0

for i, (script_name, description) in enumerate(steps, 1):
    script_path = PROJECT / script_name

    print(f"\n[{i}/{len(steps)}] {description}")
    print(f"运行: {script_name}")
    print("-" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        if result.returncode == 0:
            print(f"✅ 成功完成 ({elapsed_time:.1f}s)")
            successful += 1

            # 显示部分输出
            if result.stdout:
                lines = result.stdout.split('\n')
                # 只显示最后几行关键信息
                for line in lines[-10:]:
                    if line.strip():
                        print(f"  {line}")
        else:
            print(f"❌ 失败 ({elapsed_time:.1f}s)")
            print(f"  错误信息:")
            if result.stderr:
                print(f"  {result.stderr[:500]}")
            failed += 1

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        print(f"⏱️ 超时 ({elapsed_time:.1f}s)")
        failed += 1

    except Exception as e:
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        print(f"❌ 异常: {str(e)}")
        failed += 1

print("\n" + "=" * 80)
print("批量运行完成!")
print("=" * 80)

print(f"\n统计:")
print(f"  成功: {successful}/{len(steps)}")
print(f"  失败: {failed}/{len(steps)}")
print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

if successful == len(steps):
    print(f"\n✅ 所有脚本成功完成!")
    print(f"\n生成的图表位于:")
    print(f"  - output/step2_preprocess_abis/")
    print(f"  - output/step3_single_six_models/")
    print(f"  - output/step3_gaussian_augmentation/")
    print(f"  - output/step4_ratio_integrated_models/")
    print(f"  - output/step5_abis_bootstrap_compare/")
    print(f"  - output/step6_model_interpretation/")
    print(f"  - output/step7_final_summary/")
    print(f"  - output/step8_calibration_dca_sensitivity/")
    print(f"\n所有图表使用 Cell Metabolism biomedical palette:")
    print(f"  - Times New Roman font")
    print(f"  - PNG @ 300 DPI")
    print(f"  - Soft, cool, clean colors")
else:
    print(f"\n⚠️ 部分脚本失败,请检查错误信息并重新运行")

print("\n注意:")
print("  - 所有模型训练逻辑未改变")
print("  - 所有CSV数值保持不变")
print("  - 只更新了绘图配色方案")