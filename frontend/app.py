import streamlit as st
import requests
import json
from typing import List

st.set_page_config(
    page_title="智能技术文档问答助手",
    page_icon="📚",
    layout="wide"
)

st.title("📚 智能技术文档问答助手")
st.markdown("**技术栈**: LangChain + 通义千问 + Chroma + BGE-Reranker")

# 侧边栏
with st.sidebar:
    st.header("📊 系统状态")
    
    # 健康检查
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        st.success("✅ 服务运行正常")
    except:
        st.error("❌ 服务未启动")
    
    st.divider()
    
    st.header("📤 上传文档")
    uploaded_file = st.file_uploader(
        "选择文件",
        type=["pdf", "txt", "md", "html"]
    )
    
    if uploaded_file and st.button("上传并索引"):
        with st.spinner("处理中..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                "http://localhost:8000/api/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"失败: {response.text}")
    
    st.divider()
    
    if st.button("🗑️ 清空缓存"):
        requests.post("http://localhost:8000/api/cache/clear")
        st.success("缓存已清空")

# 主界面
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "请输入您的问题",
        placeholder="例如：LangChain的Chain如何使用？",
        key="query_input"
    )
    
    use_rerank = st.checkbox("启用Rerank重排序", value=True)
    use_cache = st.checkbox("启用语义缓存", value=True)

if query:
    with st.spinner("正在思考..."):
        try:
            response = requests.post(
                "http://localhost:8000/api/query",
                json={
                    "query": query,
                    "use_rerank": use_rerank,
                    "use_cache": use_cache
                },
                stream=True  # 流式请求
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 答案
                st.subheader("💡 答案")
                st.write(result["answer"])
                
                # 置信度
                confidence = result.get("confidence", 0)
                st.metric("置信度", f"{confidence:.2%}")
                
                # 缓存命中
                if result.get("cache_hit"):
                    st.info("📦 答案来自缓存（节省API调用）")
                
                # 代码示例
                if result.get("code_examples"):
                    st.subheader("💻 代码示例")
                    for ex in result["code_examples"]:
                        st.code(ex["code"], language=ex.get("language", "python"))
                        st.caption(ex.get("description", ""))
                
                # 参考来源
                if result.get("references"):
                    st.subheader("📖 参考来源")
                    for i, ref in enumerate(result["references"], 1):
                        with st.expander(f"来源 {i} (相关度: {ref.get('relevance_score', 0):.2f})"):
                            st.write(ref.get("content", "")[:500] + "...")
                            st.caption(f"来源: {ref.get('source', '未知')}")
            else:
                st.error(f"请求失败: {response.text}")
        
        except Exception as e:
            st.error(f"发生错误: {str(e)}")

# 底部
st.markdown("---")
st.markdown(
    "🔧 **检索策略**: 向量检索 + 关键词检索 + BGE-Rerank | "
    "💾 **缓存**: 语义缓存(相似度>0.95) | "
    "🔄 **重试**: 最多3次自动重试"
)