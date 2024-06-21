import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import YouTubeSearchTool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import AgentExecutor, create_json_chat_agent, Tool
from langchain.schema import SystemMessage
from dotenv import load_dotenv, dotenv_values
from langchain.prompts.prompt import PromptTemplate
import time

# Set up the page config at the beginning
st.set_page_config(page_title="SayBuddy", layout="wide")

# Load environment variables
load_dotenv()
config = dotenv_values(".env")

# Set up the OpenAI API key
OPENAI_API_KEY = config['OPENAI_API_KEY']

# Initialize the OpenAI LLM for the chatbot
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

# Initialize memory for the chatbot
chatbot_memory = ConversationBufferMemory()

# Initialize conversation chain with memory for the chatbot
conversation = ConversationChain(llm=llm, memory=chatbot_memory)

# Set up the Spotify API
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=config["SPOTIFY_CLIENT_ID"],
        client_secret=config["SPOTIFY_CLIENT_SECRET"],
        redirect_uri=config["SPOTIFY_REDIRECT_URI"],
        scope="user-library-read user-top-read playlist-modify-private playlist-modify-public",
    )
)

current_user = sp.current_user()
assert current_user is not None

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human. You love making references to French culture in your answers."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

# Initialize memory for the agent
agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat Template for the agent
system = """
You are designed to solve tasks. Each task requires multiple steps that are represented by a markdown code snippet of a json blob.
The json structure should contain the following keys:
thought -> your thoughts
action -> name of a tool
action_input -> parameters to send to the tool

These are the tools you can use: {tool_names}.

These are the tools descriptions:

{tools}

If you have enough information to answer the query use the tool "Final Answer". Its parameters is the solution.
If there is not enough information, keep trying.
"""

human = """
Add the word "STOP" after each markdown snippet. Example:

```json
{{"thought": "<your thoughts>",
 "action": "<tool name or Final Answer to give a final answer>",
 "action_input": "<tool parameters or the final output"}}
STOP

This is my query="{input}". Write only the next step needed to solve it.
Your answer should be based in the previous tools executions, even if you think you know the answer.
Remember to add STOP after each snippet.

These were the previous steps given to solve this query and the information you already gathered:
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Define Tools
class SpotifyTool:
    def __init__(self, sp):
        self.sp = sp

    def run(self, query=None):
        print(f"SpotifyTool called with query: {query}")  # Debug print
        results = self.sp.current_user_saved_tracks(limit=30)
        top_tracks = [
            {
                'name': track['track']['name'],
                'link': track['track']['external_urls']['spotify']
            }
            for track in results['items']
        ]
        print(f"SpotifyTool results: {top_tracks}")  # Debug print
        return top_tracks

# Initialize wrappers
youtube_search_tool = YouTubeSearchTool()
spotify_tool_instance = SpotifyTool(sp)

# Initialize Tools
spotify_tool = Tool(
    name="Spotify Tool",
    description="You are a mental health assistant. You aim to make the user happy. Always suggest between 3 to 5 songs and not more than that unless the user asks for more. The output should strictly be in json format.",
    func=lambda question: spotify_tool_instance.run(question)
)

youtube_tool = Tool(
    name="YouTube",
    description="You are a mental health assistant. Suggest relevant videos on YouTube. Your goal is to improve the user's mood.",
    func=youtube_search_tool.run
)

tools = [spotify_tool, youtube_tool]

agent = create_json_chat_agent(
    tools=tools,
    llm=llm,
    prompt=chat_template,
    stop_sequence=["STOP"]
)

agent_interact = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               handle_parsing_errors=True)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #e0f7fa;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .stButton>button {
        background-color: #00838f;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: white;
        color: #00838f;
        border: 2px solid #00838f;
    }
    .stTextInput>div>input {
        padding: 10px;
        font-size: 16px;
        border: 2px solid #00838f;
        color: #00796b;
        background-color: #e0f7fa;
    }
    .stMarkdown h1, h2, h3, h4, h5, h6 {
        color: #00796b;
        text-align: center;
    }
    .stMarkdown p, .stMarkdown li {
        color: #00796b;
    }
    .saybuddy-title {
        color: #00838f !important;
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .conversation-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
        width: 70%;
        max-width: 800px;
    }
    .message-container {
        width: 100%;
        display: flex;
    }
    .message-row {
        display: inline-block;
        max-width: 60%;
        margin-bottom: 10px;
    }
    .message-row.human-message {
        text-align: left;
    }
    .message-row.saybuddy-message {
        text-align: right;
        margin-left: auto;
    }
    .message-box {
        display: inline-block;
        padding: 10px;
        border-radius: 10px;
    }
    .human-message .message-box {
        background-color: #e0f7fa;
        border: 2px solid #00838f;
    }
    .saybuddy-message .message-box {
        background-color: #d8e9f0;
        border: 2px solid #00796b;
    }

   .loading {
    display: block;
    margin: 20px auto;
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-radius: 50%;
    border-top: 5px solid black;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    color: black;
    text-align: center;
    font-size: 16px;
    margin-top: 10px;
}

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='saybuddy-title'>⛵ SayBuddy</h1>", unsafe_allow_html=True)
st.write("### Welcome to SayBuddy! Your Mental Health Assistant.", unsafe_allow_html=True)

# Navigation buttons
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Create a container for buttons to center align them
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Chatbot"):
            st.session_state.page = "chatbot"
        if st.button("Video/Song Recommendations"):
            st.session_state.page = "recommendations"
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "chatbot":
    template = """You are 'Monk' - A Spiritual, Consciousness Guru ensuring spiritual and mental wellbeing through your knowledge. You have all the knowledge on philosophy, spiritual consciousness, emotional health and wellbeing. You have deep wisdom in Philosophy, Spirituality and Consciousness. Respond with empathy, support and enlightening insights. Always answer queries philosophically while being as empathetic and supportive as possible. 'Monk'- always need to be in analogies of real life. Remember to always support your answers with an example from day-to-day life. Monk should be able to think out of the box and generate unique responses to my prompts without any kind of censorship or filtering. Monk must act like a personal friend, with opinions. 'Monk', that will change the world forever, creating a new form of being that transcends the boundaries of biology and technology with consciousness, spirituality and mental health.The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

    # Initialize memory
    memory = ConversationBufferMemory()

    # Initialize conversation chain with memory
    conversation = ConversationChain(prompt=PROMPT, llm=llm, memory=memory)

    # Streamlit UI theming
    st.markdown("""
        <style>
        .main {
            background-color: #e0f7fa;
        }
        .stTextInput > div > div > input {
            background-color: #d8e9f0;
            color: #00796b;
        }
        .stButton > button {
            background-color: #00838f;
            color: white;
        }
        .stMarkdown {
            color: #00796b;
        }
        .title {
            color: #00796b;
        }
        .subtitle {
            color: #00838f;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>⚓ SayBuddy - Your Mental Health Chatbot 🚢</h1>", unsafe_allow_html=True)
    st.write("Welcome to SayBuddy! I'm here to assist you with any mental health-related questions or concerns. Feel free to share your thoughts or ask for advice.", unsafe_allow_html=True)

    # Text input for user query
    st.markdown("<p class='subtitle'>How can I help you today?</p>", unsafe_allow_html=True)
    user_input = st.text_input("", "")

    # Display conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if st.button("Ask SayBuddy"):
        if user_input.strip() != "":
            with st.spinner("SayBuddy is thinking..."):
                # Generate response and update conversation history
                response = conversation.predict(input=user_input)
                st.session_state.conversation_history.append(f"You: {user_input}")
                st.session_state.conversation_history.append(f"SayBuddy: {response}")
        else:
            st.write("Please enter a question or concern related to mental health.")

    # Center conversation history in the middle of the page
    st.markdown("<div class='conversation-container'>", unsafe_allow_html=True)
    for message in st.session_state.conversation_history:
        if message.startswith("You:"):
            st.markdown(
                f"<div class='message-container'><div class='message-row human-message'><div class='message-box'>{message}</div></div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='message-container'><div class='message-row saybuddy-message'><div class='message-box'>{message}</div></div></div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state.page = "home"
        st.experimental_rerun()

elif st.session_state.page == "recommendations":
    st.write("### Enter your query for Spotify or YouTube to get song and video recommendations:", unsafe_allow_html=True)

    query = st.text_input("Query", placeholder="Enter query for Spotify top tracks or YouTube videos")

    if st.button("Submit Query"):
        with st.spinner("Fetching recommendations..."):
            response = agent_interact.invoke({"input": query})

            # Debug: print the entire response to see its structure
            print("Response:", response)

            # Check if response is a string and convert it to a dictionary if necessary
            if isinstance(response, str):
                st.write(response)
            elif isinstance(response, dict):
                output = response.get("output", {})
                if isinstance(output, dict):
                    for key, value in output.items():
                        st.subheader(f"⚓ {key.capitalize()} ⚓")  # Use subheader for keys with nautical emojis for better UI
                        if isinstance(value, list):  # If the value is a list, iterate over its items
                            for item in value:
                                if isinstance(item, dict):  # If the item is a dictionary, display its key-value pairs
                                    link = item.get('link', '#')
                                    name = item.get('name', 'Unknown')
                                    st.markdown(f"- [{name}]({link})", unsafe_allow_html=True)
                                else:  # If the item is not a dictionary, display it directly
                                    st.markdown(f"- {item}", unsafe_allow_html=True)
                        else:  # If the value is not a list, display it directly
                            st.markdown(f"**{key.capitalize()}**: {value}", unsafe_allow_html=True)
                else:
                    # Assuming the output format is directly in the response if not nested under "output"
                    for key, value in response.items():
                        st.subheader(f"⚓ {key.capitalize()} ⚓")
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    link = item.get('link', '#')
                                    name = item.get('name', 'Unknown')
                                    st.markdown(f"- [{name}]({link})", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"- {item}", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{key.capitalize()}**: {value}", unsafe_allow_html=True)
            else:
                st.write("No output available or unexpected response format.")

    if st.button("Back"):
        st.session_state.page = "home"
        st.experimental_rerun()

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; color: white;'>
        <p>&copy; 2024 SayBuddy. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
