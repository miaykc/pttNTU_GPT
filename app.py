from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import streamlit as st
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain import LLMChain
from agent_prompt_template import prompt_with_history, output_parser, tools
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.memory import ConversationBufferWindowMemory

# 設定 streamlit
st.set_page_config(page_title="🦜🔗 PTT-NTU版  GPT", layout="wide")  # 設定 streamlit 網頁名稱
st.title("🦜🔗 PTT-NTU版  GPT")  # 設定 streamlit 標題

# 創建 streamlit 側邊欄位
with st.sidebar.expander("選擇 Model ", expanded=False):
    MODEL = st.selectbox(
        label="Model",
        options=[
            "gpt-3.5-turbo"
            ],
    )

# 初始化會話狀態
# 會話狀態：可以共享數據，讓應用程式記住先前的狀態
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# 使用者輸入區塊
def get_text():
    input_text = st.text_input(
        "請輸入: ",
        st.session_state["input"],
        key="input",
        placeholder="嗨！我會根據您輸入的訊息回答問題...",
    )
    return input_text

def new_chat():
    save = []  # 創建一個空列表，用於儲存聊天記錄
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])  # 存擋的格式
        save.append("Bot:" + st.session_state["generated"][i])  # 存擋的格式
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""  # 將 generated, past, input 三個變量清空，已準備開始新的聊天對話
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()
st.sidebar.button("New Chat", on_click=new_chat)

# 定義llm
llm = OpenAI(
    temperature=0.8,  # 會影響到答案的隨機性程度，因為也不是要請他寫詩或什麼，所以設低一點，回答也會比較準確
    model_name=MODEL,
    max_tokens=1500,
    verbose=True,
)
llm = OpenAI(openai_api_key="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")
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history) 

if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm)

memory=ConversationBufferWindowMemory(k=2)

# 定義 tools 和 agent
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names,
    memory=st.session_state.entity_memory
) 



# 使用者輸入
user_input = get_text()
if user_input:
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    response = agent_executor.run(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(response)

download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="👀")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    joined_download_str = "\n".join(download_str)
    if joined_download_str:
        st.download_button("Download", joined_download_str)

# 將儲存之對話記錄存到 sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# 讓使用者可以刪除所有對話
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session