import google.generativeai as genai
import os
import logging

# Configure logging
logging.basicConfig(filename='chatbot.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set. Chatbot will return error messages.")

# System prompt for e-waste assistant
SYSTEM_PROMPT = """You are an AI assistant for an E-Waste Management platform. Your role is to help users with:

1. Questions about electronic waste disposal and recycling
2. Information about device reuse, refurbishment, and donation
3. Environmental impact of electronic devices
4. Best practices for e-waste management
5. How to use the platform features

Be helpful, concise, and environmentally conscious. If users ask about specific device recommendations, encourage them to use the upload feature for personalized suggestions.

Keep responses clear and actionable. Focus on sustainability and responsible electronics management."""

def get_chatbot_response(user_message, conversation_history=None):
    """
    Get response from Gemini chatbot
    
    Args:
        user_message: The user's message
        conversation_history: List of previous messages (optional)
    
    Returns:
        The chatbot's response text
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build conversation context
        if conversation_history:
            context = SYSTEM_PROMPT + "\n\nConversation history:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                context += f"{role}: {content}\n"
            context += f"\nUser: {user_message}"
        else:
            context = SYSTEM_PROMPT + f"\n\nUser: {user_message}"
        
        # Generate response
        response = model.generate_content(context)
        return response.text
        
    except Exception as e:
        logging.error(f"Chatbot error: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again or contact support if the issue persists."

def get_quick_reply_suggestions(user_message):
    """
    Generate quick reply suggestions based on user message
    
    Returns:
        List of suggested quick replies
    """
    # Default suggestions
    default_suggestions = [
        "How do I recycle my device?",
        "What can I do with old electronics?",
        "Tell me about e-waste impact"
    ]
    
    # Context-aware suggestions
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ['recycle', 'disposal', 'throw away']):
        return [
            "Where can I recycle?",
            "What items can be recycled?",
            "Is it safe to throw in trash?"
        ]
    elif any(word in message_lower for word in ['reuse', 'donate', 'sell']):
        return [
            "Where can I donate?",
            "How to prepare device for donation?",
            "Can I sell my old device?"
        ]
    elif any(word in message_lower for word in ['phone', 'smartphone', 'mobile']):
        return [
            "How to recycle phones?",
            "Can I trade in my phone?",
            "How to wipe phone data?"
        ]
    elif any(word in message_lower for word in ['laptop', 'computer', 'pc']):
        return [
            "How to recycle computers?",
            "Can I donate my old laptop?",
            "How to remove personal data?"
        ]
    
    return default_suggestions
