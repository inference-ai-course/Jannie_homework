import requests
import json
import random
import re

def generate_response(user_text, conversation_history):
    """
    Generate a response using Llama 2 7B via Ollama.
    Falls back to intelligent responses if Ollama is not available.
    """
    # First try Ollama with Llama 2
    ollama_response = try_ollama_llama2(user_text, conversation_history)
    if ollama_response and not ollama_response.startswith("Error:"):
        return ollama_response
    
    # If Ollama fails, use intelligent fallback
    return intelligent_fallback_response(user_text, conversation_history)

def try_ollama_llama2(user_text, conversation_history):
    """
    Try to get response from Llama 2 via Ollama.
    """
    try:
        # Build conversation context
        history_text = ""
        for conv in conversation_history[-3:]:  # Last 3 exchanges for context
            if isinstance(conv, dict):
                if conv.get("user_text"):
                    history_text += f"User: {conv['user_text']}\n"
                if conv.get("assistant_text"):
                    history_text += f"Assistant: {conv['assistant_text']}\n"

        prompt = f"{history_text}User: {user_text}\nAssistant:"

        # Ollama API endpoint
        OLLAMA_URL = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama2:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 150,  # Shorter for faster voice responses
                "stop": ["User:", "###"]  # Stop sequences to prevent endless generation
            }
        }

        print(f"🔄 Sending request to Ollama Llama 2...")
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                response_text = data["response"].strip()
                # Clean up any extra conversation artifacts
                response_text = response_text.split('User:')[0].split('###')[0].strip()
                print(f"✅ Llama 2 response: {response_text[:100]}...")
                return response_text
            else:
                print("⚠️ Unexpected response format from Ollama")
                return None
        else:
            print(f"⚠️ Ollama error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama - using fallback responses")
        return None
    except requests.exceptions.Timeout:
        print("⏰ Ollama request timed out - using fallback responses")
        return None
    except Exception as e:
        print(f"❌ Error in Llama 2 response generation: {e}")
        return None

def intelligent_fallback_response(user_text, conversation_history):
    """
    Intelligent fallback responses when Llama 2 is unavailable.
    """
    user_text_lower = user_text.lower().strip()
    
    # Build recent context
    recent_context = ""
    for conv in conversation_history[-2:]:  # Last 2 exchanges
        if isinstance(conv, dict):
            if conv.get("user_text"):
                recent_context += conv['user_text'].lower() + " "
            if conv.get("assistant_text"):
                recent_context += conv['assistant_text'].lower() + " "

    # Enhanced response patterns with context awareness
    responses = {
        # Greetings
        r'^(hello|hi|hey|greetings)[\s\?!]*$': [
            "Hello! How can I assist you today?",
            "Hi there! What can I help you with?",
            "Hey! Great to hear from you. What's on your mind?",
            "Hello! I'm here and ready to help. What would you like to talk about?"
        ],
        
        # How are you
        r'how are you|how do you do|how\'s it going': [
            "I'm functioning well, thank you for asking! How can I help you today?",
            "I'm doing great! Ready to assist you. What would you like to talk about?",
            "All systems operational! How can I be of service?",
            "I'm doing well, thanks! What can I help you with today?"
        ],
        
        # Name/questions about identity
        r'who are you|what are you|your name|what is your name': [
            "I'm your voice assistant, powered by Llama 2! I'm here to help with conversations and answer your questions.",
            "I'm an AI assistant designed to chat and assist you with various topics. You can think of me as your helpful conversation partner!",
            "I'm your AI voice assistant! I can help you with questions, discussions, or just have a friendly chat.",
            "I'm here to assist you! I'm an AI assistant that can help with conversations, questions, and various topics."
        ],
        
        # Help requests
        r'help|what can you do|how to use|what can i ask': [
            "I can help you with conversations, answer questions, or just chat! What would you like to do?",
            "I'm here to assist with discussions, provide information, or simply have a friendly chat. Feel free to ask me anything!",
            "You can talk to me about anything - ask questions, share thoughts, or just converse! What's on your mind?",
            "I can help with various topics - just start a conversation, ask a question, or share what you're thinking about!"
        ],
        
        # Thanks
        r'thank|thanks|appreciate|grateful': [
            "You're welcome! Is there anything else I can help you with?",
            "My pleasure! Feel free to ask if you need anything else.",
            "Happy to help! What would you like to discuss next?",
            "You're very welcome! I'm glad I could assist you."
        ],
        
        # Goodbye
        r'bye|goodbye|see you|farewell|have a good': [
            "Goodbye! Feel free to come back if you have more questions!",
            "See you later! It was nice chatting with you.",
            "Farewell! Don't hesitate to return if you need assistance.",
            "Goodbye! Have a wonderful day!"
        ],
        
        # Questions about capabilities
        r'can you|are you able|will you|could you': [
            "I'm designed to have conversations and assist with various topics. What would you like to know?",
            "I can help with discussions and answer questions to the best of my ability! What are you curious about?",
            "I'm here to chat and provide information. What specific thing are you wondering about?",
            "I can certainly try to help! What would you like me to assist you with?"
        ]
    }

    # Context-aware follow-up responses
    if conversation_history:
        last_assistant_response = None
        for conv in reversed(conversation_history):
            if conv.get("assistant_text"):
                last_assistant_response = conv["assistant_text"].lower()
                break
        
        # Follow-up based on previous response
        if last_assistant_response:
            if any(phrase in last_assistant_response for phrase in ['how can i assist', 'what can i help', 'what would you like']):
                if not any(pattern in user_text_lower for pattern in ['hello', 'hi', 'how are you', 'thank', 'bye']):
                    follow_ups = [
                        f"I understand you mentioned: '{user_text}'. Could you tell me more about that?",
                        f"That's interesting! What more can you share about '{user_text}'?",
                        f"I see. Regarding '{user_text}', could you elaborate on that?",
                        f"Thanks for sharing that. What specifically about '{user_text}' would you like to discuss?"
                    ]
                    return random.choice(follow_ups)
            
            if 'tell me more' in last_assistant_response or 'elaborate' in last_assistant_response:
                elaboration_responses = [
                    "That's really interesting! What else would you like to share about this topic?",
                    "Thanks for the additional information! This helps me understand better.",
                    "I appreciate you expanding on that. What other aspects are you considering?",
                    "That clarifies things! Is there anything more you'd like to discuss about this?"
                ]
                return random.choice(elaboration_responses)

    # Pattern matching with regex
    for pattern, response_list in responses.items():
        if re.search(pattern, user_text_lower):
            return random.choice(response_list)
    
    # Question detection
    if user_text_lower.endswith('?'):
        question_responses = [
            "That's an interesting question. This could be approached from different perspectives.",
            "I appreciate your question. This topic often involves considering various viewpoints.",
            "That's a thoughtful question. Many people wonder about similar things.",
            "Interesting question! This reminds me of related discussions about the topic.",
            "Good question! This is something worth exploring further.",
            "That's a great question. Let me share some thoughts on this topic."
        ]
        return random.choice(question_responses)
    
    # Statement detection based on length
    word_count = len(user_text.split())
    
    if word_count > 10:
        # Longer, more detailed statement
        long_responses = [
            "I see what you're saying. That's quite insightful and gives me a better understanding of your perspective.",
            "Thank you for sharing that detailed explanation. It helps me understand your viewpoint better.",
            "That's a comprehensive point. I appreciate you taking the time to explain it so clearly.",
            "I understand your perspective. That's a well-articulated point you've made.",
            "Thanks for explaining that so thoroughly. It really helps me grasp what you're thinking."
        ]
        return random.choice(long_responses)
    
    elif word_count > 5:
        # Medium-length statement
        medium_responses = [
            "I understand what you're saying. That's an interesting perspective.",
            "That's a good point. It reminds me of related discussions.",
            "Thanks for sharing that with me. What would you like to discuss next?",
            "I see what you mean. That's something worth considering.",
            "That's an interesting observation. What are your thoughts on this?"
        ]
        return random.choice(medium_responses)
    
    else:
        # Short statement or fragment
        short_responses = [
            "I understand.",
            "That's interesting.",
            "Thanks for sharing.",
            "I see what you mean.",
            "That makes sense.",
            "Got it.",
            "Interesting point."
        ]
        return random.choice(short_responses)

# Test function
if __name__ == "__main__":
    # Test the fallback responses
    test_conversations = [
        {"user_text": "Hello", "assistant_text": "Hi there!"},
        {"user_text": "How are you?", "assistant_text": "I'm doing well!"}
    ]
    
    test_inputs = [
        "Hello",
        "How are you?",
        "What can you do?",
        "Thank you",
        "What do you think about AI?",
        "I've been thinking about learning programming"
    ]
    
    print("🧪 Testing Llama 2 fallback responses:\n")
    for test_input in test_inputs:
        response = generate_response(test_input, test_conversations)
        print(f"Input: '{test_input}'")
        print(f"Response: '{response}'\n")