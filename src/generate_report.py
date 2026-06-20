"""
Generate report.pdf in Chinese following the course template format.
"""
import os
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Font paths
FONT_DIR = "C:/Windows/Fonts"
SIMSUN = os.path.join(FONT_DIR, "simsun.ttc")
SIMHEI = os.path.join(FONT_DIR, "simhei.ttf")
SIMKAI = os.path.join(FONT_DIR, "simkai.ttf")


class ChineseReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("SimSun", "", SIMSUN)
        self.add_font("SimHei", "", SIMHEI)
        self.add_font("SimKai", "", SIMKAI)

    def header(self):
        self.set_font("SimHei", "", 18)
        self.cell(0, 10, "2026 年人工智能导论大作业", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("SimSun", "", 14)
        self.cell(0, 8, "（可解释的谣言检测）", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("SimSun", "", 9)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    def section_title(self, title, num=""):
        self.ln(3)
        self.set_font("SimHei", "", 13)
        full = f"{num} {title}" if num else title
        self.cell(0, 8, full, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("SimHei", "", 11)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("SimSun", "", 10.5)
        self.multi_cell(0, 6, text, align="L")
        self.ln(1)

    def bullet(self, text):
        self.set_font("SimSun", "", 10.5)
        x = self.get_x()
        self.cell(5)
        self.multi_cell(0, 6, f"● {text}", align="L")
        self.ln(0.5)

    def info_line(self, label, value=""):
        self.set_font("SimSun", "", 11)
        self.cell(25, 7, label)
        self.set_font("SimKai", "", 11)
        self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")


def generate_report():
    pdf = ChineseReport()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Info section ──
    pdf.ln(4)
    pdf.info_line("姓  名：", "（填写姓名）")
    pdf.info_line("学  号：", "（填写学号）")
    pdf.info_line("班  级：", "（填写班级）")
    pdf.info_line("日  期：", "2026 年 6 月 20 日")
    pdf.ln(6)

    # ═══════════════════════════════════════════════
    # 1. 问题描述
    # ═══════════════════════════════════════════════
    pdf.section_title("问题描述", "一、")

    pdf.body(
        "谣言检测是社交媒体信息治理中的关键任务。本作业要求构建一个可解释的谣言检测模型，"
        "该模型不仅需要对推文进行二分类（0 表示非谣言，1 表示谣言），还需输出文字形式的判断依据，"
        "说明分类的推理过程。数据集包含 2,840 条训练样本和 401 条验证样本，涵盖 7 个不同事件类别。"
        "模型需兼顾分类准确率、可解释性、泛化能力以及合理的运行时间。"
    )

    # ═══════════════════════════════════════════════
    # 2. 模型设计
    # ═══════════════════════════════════════════════
    pdf.section_title("模型设计", "二、")

    # (1) 整体架构
    pdf.sub_title("（1）整体架构")
    pdf.body(
        "本系统采用深度学习 + 多策略解释的复合架构。分类模块使用预训练语言模型 DistilBERT "
        "进行微调，输出二分类标签与置信度；解释模块结合梯度归因、语言学模式分析与模板化自然语言生成，"
        "将模型内部决策信号转换为人类可读的判断依据。整体流程为：输入文本 → 文本清洗 → "
        "DistilBERT 编码与分类 → 梯度归因分析 → 语言学特征检测 → 解释文本合成 → 输出。"
    )

    # (2) 分类模型设计
    pdf.sub_title("（2）分类模型设计")
    pdf.body(
        "分类模块基于 distilbert-base-uncased 构建。DistilBERT 是 BERT 的知识蒸馏版本，"
        "保留了 97% 的性能，参数量减少 40%，推理速度提升 60%，适合在 CPU 环境下运行。"
        "在预训练模型之上添加一个二分类线性头，使用交叉熵损失进行端到端微调。"
    )
    pdf.body(
        "主要超参数设置：最大序列长度 128，批大小 16，学习率 5e-5，优化器 AdamW，"
        "训练轮数 10，采用线性预热调度策略（预热比例 0.1）。训练过程中在验证集上评估准确率，"
        "保留最佳模型以防过拟合。"
    )

    # (3) 解释模块设计
    pdf.sub_title("（3）解释模块设计（可解释性）")
    pdf.body(
        "解释模块采用三种互补策略生成判断依据："
    )
    pdf.body(
        "① 梯度归因分析：对预测类别的 logit 反向传播计算词嵌入梯度，以梯度范数衡量各 token "
        "对决策的贡献度，识别出模型重点关注的关键词语。"
    )
    pdf.body(
        "② 语言学模式检测：基于正则表达式匹配谣言和非谣言的语言特征。谣言特征包括："
        "突发新闻用语（BREAKING）、未经验证的说法（allegedly）、情绪化表达（shocking）、"
        "隐瞒指控（cover-up）、号召性用语（wake up）等；非谣言特征包括：明确信源引用"
        "（according to）、官方确认用语（confirmed）、研究数据引用等。"
    )
    pdf.body(
        "③ 模板化自然语言生成：将归因结果和语言学分析发现组织成连贯的解释段落，"
        "包含预测结果、置信度、关键信号词和推理逻辑。"
    )

    # ═══════════════════════════════════════════════
    # 3. 实验结果
    # ═══════════════════════════════════════════════
    pdf.section_title("实验结果", "三、")

    # (1) 实验设置
    pdf.sub_title("（1）实验设置")
    pdf.body(
        "实验环境：Python 3.13，PyTorch 2.12（CPU），transformers 4.x。"
        "训练集 2,840 条（非谣言 1,600 条，谣言 1,240 条），验证集 401 条。"
        "评估指标包括准确率（Accuracy）、精确率（Precision）、召回率（Recall）和 F1 值。"
    )

    # (2) 模型性能与分析
    pdf.sub_title("（2）模型性能与分析")
    pdf.body(
        "验证集评估结果如下表所示："
    )
    pdf.ln(1)

    # Simple table
    pdf.set_font("SimHei", "", 10)
    col_w = [38, 28, 30, 30, 30]
    headers = ["类别", "精确率", "召回率", "F1 值", "样本数"]
    pdf.set_fill_color(230, 230, 230)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("SimSun", "", 10)
    data = [
        ["非谣言", "0.86", "0.89", "0.88", "226"],
        ["谣言", "0.85", "0.82", "0.83", "175"],
        ["加权平均", "0.86", "0.86", "0.86", "401"],
    ]
    for row in data:
        for i, cell in enumerate(row):
            pdf.cell(col_w[i], 7, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(2)
    pdf.body(f"整体准确率为 85.79%（344/401）。混淆矩阵显示：非谣言误判为谣言 25 例，谣言误判为非谣言 32 例。")
    pdf.ln(1)

    pdf.body(
        "分析：(1) 模型在两类上表现均衡，非谣言类的召回率略高（0.89），说明模型识别非谣言较为保守；"
        "(2) 谣言类的 F1 值达 0.83，表明模型能有效捕捉谣言的语言模式；"
        "(3) 模型在 CPU 环境下单条推理时间不足 1 秒，满足实时性要求；"
        "(4) 解释模块能准确识别文本中的关键信号词和语言模式，生成的判断依据具有较好的可读性和说服力。"
    )

    # ═══════════════════════════════════════════════
    # 4. 总结
    # ═══════════════════════════════════════════════
    pdf.section_title("总结", "四、")

    pdf.body(
        "本项目成功实现了一个可解释的谣言检测系统。系统采用 DistilBERT 作为分类骨干，"
        "结合梯度归因、语言学模式分析和模板生成三种策略，同时给出分类结果和判断依据。"
        "在验证集上取得 85.79% 的准确率和 0.86 的加权 F1 值，推理速度满足实用需求。"
        "模型对 7 个不同事件类别的推文具有良好的泛化能力。"
    )

    pdf.body(
        "不足之处：(1) 语言学模式依赖人工规则，覆盖范围有限；"
        "(2) 缺少解释质量的自动化评估指标；(3) 未利用用户画像、传播网络等多模态信息。"
    )

    pdf.body(
        "未来可尝试引入更大规模的语言模型（如 LLaMA、ChatGPT 等）来提升解释的自然度和丰富度，"
        "同时可利用注意力可视化增强可解释性的直观性。"
    )

    # Save
    output_path = os.path.join(BASE_DIR, "report.pdf")
    pdf.output(output_path)
    print(f"报告已保存至: {output_path}")


if __name__ == "__main__":
    generate_report()
