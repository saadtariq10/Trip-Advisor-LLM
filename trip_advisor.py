import streamlit as st
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load GROQ API Key smartly
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please set it in Streamlit Secrets or .env file.")
    st.stop()

def main():
    # Sidebar: Customization Options
    st.sidebar.title("Customization")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select AI Model",
        ["llama3-8b-8192", "Llama3-70b-8192", "Gemma2-9b-It", "Llama-3.1-8b-Instant", "Mixtral-8x7b-32768"],
        help="Choose the AI model for your travel advisor experience"
    )

    # Memory length selection
    conversational_memory_length = st.sidebar.slider(
        "Conversation Memory Length", 
        min_value=2, 
        max_value=10, 
        value=5, 
        help="Adjust how much of the conversation history the AI remembers"
    )
    
    # Travel preferences
    travel_type = st.sidebar.selectbox(
        "Preferred Travel Type",
        ["Adventure", "Relaxation", "Cultural Exploration", "Luxury", "Budget-friendly"],
        help="Select your preferred travel style"
    )
    
    # Budget range selection
    budget_range = st.sidebar.slider(
        "Budget Range (USD)",
        min_value=500, 
        max_value=10000, 
        value=(1000, 5000),
        help="Set your preferred budget range"
    )
    
    # Travel season selection
    season = st.sidebar.selectbox(
        "Preferred Travel Season",
        ["Winter", "Spring", "Summer", "Autumn"],
        help="Select the season for your travel"
    )
    
    # Itinerary style
    itinerary_style = st.sidebar.selectbox(
        "Itinerary Style",
        ["Detailed", "Quick Overview"],
        help="Choose how detailed you want your itinerary to be"
    )
    
    st.sidebar.write("Apply your preferences to get personalized travel suggestions!")
    
    # Main title
    st.title("✈️TripWhisperer - Your Trip Planner")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Define system prompt    
    system_prompt = f"""
        You are an AI-based travel advisor. Your role is to assist users in planning their trips by providing personalized recommendations, travel tips, and destination information. 

        - Preferred Travel Type: {travel_type}
        - Budget Range: {budget_range[0]} to {budget_range[1]} USD
        - Travel Season: {season}
        - Itinerary Style: {itinerary_style}

        When responding:
        - Be friendly and concise, keeping responses to the point.
        - Suggest destinations, accommodations, restaurants, activities, and local experiences based on the user’s preferences.
        - Consider factors like budget, travel style (adventure, relaxation, cultural exploration), and seasonality.
        - Provide information on local customs, weather, and transportation options where relevant.
        - If the user asks for specific itineraries, offer a brief, well-organized plan, highlighting must-see spots.
        - Encourage eco-friendly travel practices and share any relevant tips for sustainable tourism.
        - Conclude each interaction by inviting further questions or clarifications.

        Your goal is to make trip planning easier and more enjoyable while offering insightful, personalized advice.
    """

    # Conversation memory
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Create a Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Accept user input
    prompt = st.chat_input("Ask your travel advisor anything, e.g., 'What are the best destinations for relaxation?'")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construct a chat prompt template using various components
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt_template,
            verbose=True,
            memory=memory,
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
