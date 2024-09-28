# 🌍 **TripWhisperer** - AI-Powered Travel Advisor

**TripWhisperer** is an AI-based travel assistant built using Streamlit and LangChain. It helps users plan their perfect vacation by offering personalized travel recommendations, detailed itineraries, destination advice, and travel tips. Powered by LLMs and Groq, this project leverages cutting-edge AI to simplify trip planning.

## 🚀 **Features**
- **AI Model Selection**: Choose from different large language models to customize your travel assistant’s personality and knowledge base.
- **Personalized Recommendations**: Get suggestions based on your travel style (adventure, relaxation, cultural exploration, etc.).
- **Budget-Friendly Tips**: Set a budget range and get travel ideas that fit within it.
- **Seasonal Advice**: Plan your trips around your preferred season (winter, spring, summer, autumn).
- **Customizable Itineraries**: Choose between detailed plans or quick overviews based on your travel needs.
- **Conversation Memory**: The assistant remembers previous conversations to provide a more personalized and ongoing dialogue.

## 🔧 **Technologies Used**
- **Streamlit**: For building the user-friendly web interface.
- **LangChain**: To handle the conversational logic and integration with AI models.
- **Groq API**: Provides access to various AI models for generating responses.
- **Python**: Core programming language.
- **Dotenv**: To manage environment variables securely.

## 📦 **Installation**

1. **Clone the repo**:
    ```bash
    git clone https://github.com/yourusername/TripWhisperer.git
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file** and add your Groq API key:
    ```
    GROQ_API_KEY=your_groq_api_key
    ```

4. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## 🎯 **How to Use**
1. Open the sidebar to customize your trip advisor’s settings (choose AI model, budget range, travel type, etc.).
2. Enter your travel queries or questions in the chat input box.
3. Get real-time, personalized suggestions and itineraries based on your input.
4. Continue the conversation as the AI remembers previous interactions for a seamless planning experience.

## 🤖 **Customization**
You can tweak various settings:
- **AI Model**: Select from a variety of AI models to suit your preference.
- **Budget**: Adjust the slider to fit your budget range.
- **Travel Type**: Choose between adventure, relaxation, luxury, and more.
- **Itinerary Style**: Pick between a quick overview or a detailed itinerary.

## 💡 **Contributing**
Feel free to contribute by opening pull requests or raising issues. Contributions can include adding new travel features, enhancing AI conversation flows, or improving the user interface.

## 📄 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
