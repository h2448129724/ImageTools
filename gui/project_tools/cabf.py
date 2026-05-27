"""CAB-F project tool registration."""
from __future__ import annotations

from gui.project_tools.models import RegisteredProject, RegisteredProjectTool


PROJECT = RegisteredProject(
    key="cabf",
    menu_title="CAB-F",
    display_name="CAB-F",
    tools=(
        RegisteredProjectTool(
            key="cabf_stitch_editor",
            title="连边标注器",
            description="编辑 CAB-F 图像中的缝纫点与连边关系。若当前主界面已经加载图片，会自动把当前图片带入编辑器。",
            launcher_name="_show_stitch_point_editor",
        ),
        RegisteredProjectTool(
            key="cabf_point_filter",
            title="缝纫点数据筛选",
            description="筛选和复核 CAB-F 缝纫点数据，适合批量检查异常样本、脏数据和候选修复对象。",
            launcher_name="_show_stitch_point_filter",
        ),
        RegisteredProjectTool(
            key="cabf_dataset_tool",
            title="数据集校验与导出",
            description="校验 CAB-F 母数据，按样本汇总问题，并导出训练数据到 images / annotations / error 目录结构。",
            launcher_name="_show_cabf_dataset_tool",
        ),
    ),
)
