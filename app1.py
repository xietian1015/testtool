import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import traceback

# 尝试导入 fitparse，如果失败则提供替代方案
try:
    from fitparse import FitFile
    FITPARSE_AVAILABLE = True
except ImportError:
    FITPARSE_AVAILABLE = False
    st.warning("fitparse 库不可用，FIT 文件处理功能将受限")

st.set_page_config(page_title="跑步数据精提取工具 v3.2", layout="wide")

st.title("跑步数据精提取工具 v3.2")

# 显示警告如果 fitparse 不可用
if not FITPARSE_AVAILABLE:
    st.error("""
    ⚠️ FIT 文件处理功能不可用。 
    请确保在 requirements.txt 中包含 fitparse 库。
    当前只能处理 CSV 文件。
    """)

# 文件上传区域
col1, col2 = st.columns(2)
with col1:
    csv_file = st.file_uploader("上传CSV文件", type=["csv"])
with col2:
    if FITPARSE_AVAILABLE:
        fit_file = st.file_uploader("上传FIT文件", type=["fit"])
    else:
        st.info("FIT 文件上传不可用")

# 处理按钮
col1, col2 = st.columns(2)
with col1:
    process_btn = st.button("本组数据处理", disabled=(csv_file is None))
with col2:
    compare_btn = st.button("CSV vs FIT 对比 (HAR/HAPE)", 
                           disabled=(csv_file is None or (FITPARSE_AVAILABLE and fit_file is None) or not FITPARSE_AVAILABLE))

# 进度条
progress_bar = st.progress(0)

def read_gomore_csv(uploaded_file):
    """读取Gomore格式的CSV文件"""
    try:
        data_lines = []
        header = None
        version_info = None
        
        # 读取文件内容
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        # 处理第一行（可能包含版本信息）
        if "Gomore version" in lines[0] or "Conore version" in lines[0]:
            version_info = lines[0].strip()
            # 第二行应该是列名
            if len(lines) > 1:
                header = lines[1].strip().split('\t')
                start_line = 2
            else:
                raise ValueError("文件格式不正确")
        else:
            # 第一行就是列名
            header = lines[0].strip().split('\t')
            start_line = 1
        
        # 处理数据行
        for i in range(start_line, len(lines)):
            line = lines[i].strip()
            if not line:  # 跳过空行
                continue
                
            values = line.split('\t')
            # 确保每行的列数与标题一致
            if len(values) == len(header):
                data_lines.append(values)
            else:
                # 如果列数不匹配，尝试修复
                if len(values) > len(header):
                    values = values[:len(header)]
                else:
                    values.extend([''] * (len(header) - len(values)))
                data_lines.append(values)
        
        # 创建DataFrame
        df = pd.DataFrame(data_lines, columns=header)
        
        # 记录版本信息
        if version_info:
            st.sidebar.info(f"CSV版本信息: {version_info}")
        
        return df
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        st.text(traceback.format_exc())
        return pd.DataFrame()

def parse_fit(uploaded_file):
    """解析FIT文件"""
    if not FITPARSE_AVAILABLE:
        st.error("fitparse 库不可用，无法解析 FIT 文件")
        return pd.DataFrame()
    
    try:
        # 将上传的文件保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fit') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        fitfile = FitFile(tmp_path)
        records = []
        for record in fitfile.get_messages("record"):
            row = {}
            for data in record:
                row[data.name] = data.value
            records.append(row)
        
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"解析FIT文件时出错: {str(e)}")
        st.text(traceback.format_exc())
        # 确保清理临时文件
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return pd.DataFrame()

def process_data():
    """处理CSV数据"""
    try:
        progress_bar.progress(10)
        df = read_gomore_csv(csv_file)
        progress_bar.progress(50)
        
        if df.empty:
            return
        
        # 显示文件基本信息
        st.subheader("CSV文件基本信息")
        st.write(f"行数: {len(df)}")
        st.write(f"列数: {len(df.columns)}")
        
        # 尝试转换数值列
        numeric_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():  # 如果列中有数值数据
                    numeric_cols.append(col)
            except:
                pass
        
        if len(numeric_cols) > 0:
            st.subheader("数值列统计信息")
            stats = df[numeric_cols].describe().T
            st.dataframe(stats)
            
            # 特别输出一些关键指标
            key_metrics = ['cadence', 'stepCnt', 'pace', 'speedOut', 'hrRef']
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if available_metrics:
                st.subheader("关键指标统计")
                for metric in available_metrics:
                    if not df[metric].isna().all():
                        st.write(f"{metric}: 均值={df[metric].mean():.2f}, 标准差={df[metric].std():.2f}")
        else:
            st.warning("没有找到数值列")
        
        progress_bar.progress(100)
        st.success("数据处理完成")
    except Exception as e:
        st.error(f"处理数据时出错: {str(e)}")
        st.text(traceback.format_exc())

def compare_data():
    """对比CSV与FIT数据"""
    if not FITPARSE_AVAILABLE:
        st.error("fitparse 库不可用，无法进行对比")
        return
        
    try:
        progress_bar.progress(20)
        df_csv = read_gomore_csv(csv_file)
        progress_bar.progress(40)
        
        if df_csv.empty:
            return
        
        df_fit = parse_fit(fit_file)
        progress_bar.progress(60)
        
        if df_fit.empty:
            return
        
        # 显示文件基本信息
        st.subheader("文件基本信息")
        st.write(f"CSV文件行数: {len(df_csv)}")
        st.write(f"FIT文件行数: {len(df_fit)}")
        
        # 尝试找到共同的列进行比较
        common_columns = set(df_csv.columns).intersection(set(df_fit.columns))
        if not common_columns:
            st.warning("CSV和FIT文件没有共同的列名")
            return
        
        st.subheader("共同列对比结果")
        st.write(f"共同列: {', '.join(common_columns)}")
        
        # 对每个共同列进行比较
        for col in common_columns:
            if col in df_csv.columns and col in df_fit.columns:
                try:
                    csv_data = pd.to_numeric(df_csv[col].dropna(), errors='coerce')
                    fit_data = pd.to_numeric(df_fit[col].dropna(), errors='coerce')
                    
                    # 移除NaN值
                    mask = ~(np.isnan(csv_data) | np.isnan(fit_data))
                    csv_data = csv_data[mask]
                    fit_data = fit_data[mask]
                    
                    # 对齐长度
                    n = min(len(csv_data), len(fit_data))
                    if n == 0:
                        st.write(f"列 '{col}' 没有有效数值数据")
                        continue
                        
                    csv_data = csv_data[:n]
                    fit_data = fit_data[:n]

                    # 计算HAR和HAPE
                    har = np.mean(np.abs(csv_data - fit_data))
                    hape = np.mean(np.abs((csv_data - fit_data) / fit_data)) * 100

                    st.write(f"列 '{col}' 对比 -> HAR: {har:.2f}, HAPE: {hape:.2f}%")
                except Exception as e:
                    st.write(f"无法比较列 '{col}': {e}")
        
        progress_bar.progress(100)
        st.success("数据对比完成")
    except Exception as e:
        st.error(f"对比数据时出错: {str(e)}")
        st.text(traceback.format_exc())

# 处理按钮点击事件
if process_btn:
    process_data()

if compare_btn:
    compare_data()

# 添加使用说明
st.sidebar.header("使用说明")
st.sidebar.info("""
1. 上传您的跑步数据CSV文件
2. 上传对应的FIT文件（可选）
3. 点击"本组数据处理"分析CSV文件
4. 点击"CSV vs FIT 对比"比较两种格式的数据
""")

# 显示环境信息（用于调试）
st.sidebar.header("环境信息")
st.sidebar.text(f"Pandas版本: {pd.__version__}")
st.sidebar.text(f"Numpy版本: {np.__version__}")
st.sidebar.text(f"Fitparse可用: {FITPARSE_AVAILABLE}")
#test