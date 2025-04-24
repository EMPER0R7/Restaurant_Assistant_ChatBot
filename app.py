import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import time
from collections import deque
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple, Optional
import random


MAX_HISTORY_LENGTH = 10  
DEFAULT_K = 5  
TEMPERATURE = 0.3  


st.set_page_config(
    page_title="ZOMO Restaurant Chatbot",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(""" 
<style> 
    /* Core app styling */
    .stApp { 
        max-width: 1200px; /* Increased width */
        margin: 0 auto; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #fafafa; 
    } 
    
    /* Chat message styling with animation */
    .chat-message { 
        padding: 1.25rem; 
        margin-bottom: 1rem; 
        border-radius: 0; 
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border: none;
        animation: fadeIn 0.3s ease;
        transition: all 0.2s ease;
    } 
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message { 
        background-image: linear-gradient(to right, #f0f0f0, #f8f8f8); 
        border-left: 3px solid #333; 
    } 
    .bot-message { 
        background-image: linear-gradient(to right, #ffffff, #f9f9f9); 
        border-left: 3px solid #555; 
    } 
    .chat-message:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Input field styling */
    .stTextInput > div > input { 
        border: 1px solid #e0e0e0; 
        border-radius: 0;
        padding: 0.875rem; 
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: linear-gradient(to right, #fafafa, #ffffff);
    } 
    .stTextInput > div > input:focus {
        border-color: #333;
        box-shadow: none;
        background: #ffffff;
    }
    
    /* Button styling with gradient and animation */
    .stButton > button { 
        background: linear-gradient(135deg, #333 0%, #555 100%); 
        color: white; 
        border-radius: 0; 
        padding: 0.5rem 1.25rem; 
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    } 
    .stButton > button:hover { 
        background: linear-gradient(135deg, #444 0%, #666 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    } 
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Typography with subtle animation */
    .stMarkdown h1 { 
        color: #333;
        font-weight: 600;
        font-size: 1.75rem;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #333, #555);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.3s ease;
    }
    .stMarkdown h1:hover {
        letter-spacing: -0.3px;
    }
    .stMarkdown h2 {
        font-weight: 500;
        letter-spacing: -0.3px;
        font-size: 1.4rem;
    }
    
    
    /* Restaurant card styling with gradient and animation */
    .restaurant-card { 
        padding: 1.5rem; 
        border-radius: 0; 
        margin-bottom: 1.5rem; 
        border: none; 
        background: linear-gradient(145deg, #ffffff, #f9f9f9);
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .restaurant-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
    }
    
    /* Custom elements */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #ddd, transparent);
    }
    
    /* Remove streamlit component borders */
    div.block-container {
        padding-top: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Subtle page loading animation */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #333, #555, #777, #555, #333);
        background-size: 400% 400%;
        animation: gradientBG 3s ease infinite;
        z-index: 1000;
    }
    
    /* Hide default Streamlit menu and footer */
    #MainMenu, footer {
        visibility: hidden;
    }
</style> 
""", unsafe_allow_html=True)


def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = deque(maxlen=MAX_HISTORY_LENGTH)
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "run_query" not in st.session_state:
        st.session_state.run_query = False
    if "processing" not in st.session_state:
        st.session_state.processing = False

initialize_session_state()


@st.cache_resource(show_spinner="Loading restaurant data...")
def load_data():
    try:
        df = pd.read_csv("zomato_restaurants_final_fixed.csv", encoding='utf-8-sig').fillna("")
        
        # Clean price information
        def fix_price(p):
            try:
                return int(''.join([ch for ch in str(p) if ch.isdigit()]) or 0)
            except:
                return 0
        
        df['clean_price'] = df['Price'].apply(fix_price)
        
     
        df['text'] = df.apply(lambda row: (
            f"Menu Item: {row['Item Name']}\n"
            f"Restaurant: {row['Name']}\n"
            f"Price: ₹{row['clean_price']}\n"
            f"Category: {row['Category']}\n"
            f"Description: {row['Description']}\n"
            f"Diet: {'Vegetarian' if row['Vegetarian'].strip().lower() == 'yes' else 'Non-Vegetarian'}\n"
            f"Cuisines: {row['Cuisines']}"
        ), axis=1)
        
       
        df['is_vegetarian'] = df['Vegetarian'].str.strip().str.lower() == 'yes'
        df['cuisine_list'] = df['Cuisines'].str.split(',').apply(lambda x: [c.strip() for c in x] if isinstance(x, list) else [])
        
        restaurants = sorted(df['Name'].unique())
        categories = sorted(df['Category'].unique())
        cuisine_types = sorted(set([c for sublist in df['cuisine_list'] for c in sublist if c]))
        
        return df, df['text'].tolist(), restaurants, categories, cuisine_types
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), [], [], [], []


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

@st.cache_resource(show_spinner="Building vector index...")
def build_faiss_index(embeddings: np.ndarray):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Error building FAISS index: {str(e)}")
        return None

@st.cache_resource(show_spinner="Loading generator model...")
def load_generator():
    try:
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base"),
            device="cpu",
            temperature=TEMPERATURE
        )
    except Exception as e:
        st.error(f"Error loading generator model: {str(e)}")
        return None


@st.cache_resource(show_spinner="Loading intent classifier...")
def load_intent_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

intent_classifier = load_intent_classifier()


def extract_entities(query: str) -> Dict:
    """Enhanced entity extraction with better pattern matching"""
    query_lower = query.lower()
    entities = {
        "restaurants": [],
        "categories": [],
        "cuisines": [],
        "price_range": None,
        "dietary": None,
        "comparison": False
    }
    
 
    for restaurant in all_restaurants:
         if any(rest_part.lower() in query_lower for rest_part in restaurant.split()):
            if restaurant not in entities["restaurants"]:
                entities["restaurants"].append(restaurant)
    
    for category in all_categories:
       category_lower = category.lower()
       if category_lower in query_lower or any(
            cat_part in query_lower for cat_part in category_lower.split() if len(cat_part) > 3
        ):
            entities["categories"].append(category)
    
    for cuisine in all_cuisine_types:
        if re.search(r'\b' + re.escape(cuisine.lower()) + r'\b', query_lower):
            entities["cuisines"].append(cuisine)
    
   
    price_patterns = [
        r"(?:under|below|less than|cheaper than|up to|maximum|max)\s*₹?\s*(\d+)",
        r"₹?\s*(\d+)\s*(?:and under|or less|or below)",
        r"(\d+)\s*rupees?",
        r"(\d+)\s*rs"
    ]
    for pattern in price_patterns:
        if match := re.search(pattern, query_lower):
            entities["price_range"] = int(match.group(1))
            break
    budget_terms = ["cheap", "budget", "affordable", "inexpensive", "economical"]
    if any(term in query_lower for term in budget_terms) and not entities["price_range"]:
        entities["price_range"] = 500

    # Dietary preferences
    diet_map = {
        "vegetarian": "vegetarian",
        "veg": "vegetarian",
        "vegan": "vegetarian",
        "non-veg": "non-vegetarian",
        "non vegetarian": "non-vegetarian",
        "gluten-free": "gluten-free",
        "gluten free": "gluten-free"
    }
    for term, diet in diet_map.items():
        if term in query_lower:
            entities["dietary"] = diet
            break

    veg_terms = ["veg", "vegetarian", "plant", "no meat", "no chicken", "no fish"]
    nonveg_terms = ["non veg", "non-veg", "nonveg", "chicken", "meat", "fish", "beef", "mutton"]    
    
    if any(term in query_lower for term in veg_terms):
        entities["dietary"] = "vegetarian"
    elif any(term in query_lower for term in nonveg_terms):
        entities["dietary"] = "non-vegetarian"
    
  
    if any(term in query_lower for term in ["spicy", "hot", "chili"]):
        entities["spice_level"] = "spicy"
    elif any(term in query_lower for term in ["mild", "not spicy", "less spicy"]):
        entities["spice_level"] = "mild"
    
    return entities
    # Comparison detection
    entities["comparison"] = any(
        word in query_lower
        for word in ["compare", "versus", "vs", "difference between", "which is better"]
    ) and len(entities["restaurants"]) >= 2
    
    return entities

def filter_dataframe(df: pd.DataFrame, entities: Dict) -> pd.DataFrame:
    """Enhanced dataframe filtering with better cuisine handling"""
    filtered = df.copy()
    
  
    if entities["restaurants"]:
        filtered = filtered[filtered['Name'].isin(entities["restaurants"])]
    
 
    if entities["categories"]:
        filtered = filtered[filtered['Category'].isin(entities["categories"])]
    

    if entities["cuisines"]:
        cuisine_filter = filtered['cuisine_list'].apply(
            lambda x: any(c.lower() in [cl.lower() for cl in entities["cuisines"]] for c in x)
        )
        filtered = filtered[cuisine_filter]
    
   
    if entities["price_range"] is not None:
        filtered = filtered[filtered['clean_price'] <= entities["price_range"]]
    
 
    if entities["dietary"] == "vegetarian":
        filtered = filtered[filtered['is_vegetarian'] == True]
    elif entities["dietary"] == "non-vegetarian":
        filtered = filtered[filtered['is_vegetarian'] == False]
    elif entities["dietary"] == "gluten-free":
        filtered = filtered[filtered['Description'].str.contains('gluten-free', case=False, na=False)]
    
    return filtered

def retrieve_chunks(query: str, k: int = DEFAULT_K) -> List[str]:
    """Enhanced retrieval with fallback logic"""
    try:
        query_embedding = embed_model.encode([query], normalize_embeddings=True)
        distances, indices = index.search(query_embedding.astype("float32"), k)
        return [text_chunks[i] for i in indices[0] if i >= 0]
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return []

def generate_response(query: str, context: str, conversation_history: List[Tuple[str, str]]) -> str:
    """Improved response generation with conversational tone"""
    prompt = f"""You are a friendly restaurant expert assistant. Answer the user's question conversationally but professionally based on the context.

Previous conversation:
{format_conversation_history(conversation_history[-3:])}

Relevant context:
{context}

User's question: {query}

Guidelines:
- Be helpful and polite
- Use natural language like a knowledgeable friend
- When mentioning prices, always include the ₹ symbol
- For dietary options, be explicit (e.g., "This is vegetarian")
- If you're unsure, admit it politely
- Keep responses concise but informative

Response:"""
    
    try:
        response = generator(
            prompt,
            max_length=300,
            do_sample=True,
            temperature=0.5,  
            num_beams=3
        )[0]['generated_text']

        if "guidelines:" in response.lower() or "if you're unsure" in response.lower():
            response = response.split("guidelines:", 1)[0].strip()
        
        
        response = re.sub(r'(\d+) rupees', '₹\1', response)
        response = re.sub(r'(\d+) rs', '₹\1', response)
        
       
        if "sorry" in response.lower() or "don't know" in response.lower():
            response = random.choice([
                "I couldn't find definitive information about that. Could you try rephrasing?",
                "I'm not entirely sure about that one. Maybe ask about something else on the menu?",
                "Hmm, I don't have enough details to answer that precisely. Try asking about specific restaurants or dishes."
            ])
        
        return response.strip()
    except Exception as e:
        return "Sorry, I encountered an error. Please try again."

def format_conversation_history(history: List[Tuple[str, str]]) -> str:
    """Format conversation history for context"""
    return "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {message}"
        for role, message in history
    )

def handle_greeting(query: str) -> Optional[str]:
    """Handle greetings and introductory questions"""
    greeting_keywords = ["hii", "hello", "hey", "greetings", "good morning", "good afternoon"]
    intro_keywords = ["who are you", "what can you do", "your purpose", "help"]
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in greeting_keywords):
        return random.choice([
            "Hello! 👋 I'm your restaurant assistant. How can I help you today?",
            "Hi there! 🍽️ Ask me anything about restaurants and menus.",
            "Greetings! I can help you find food options. What would you like to know?"
        ])
    
    if any(word in query_lower for word in intro_keywords):
        return (
            "I'm an AI restaurant assistant powered by Zomato data. I can help you:\n"
            "- Find menu items by price, diet, or cuisine\n"
            "- Compare restaurants\n"
            "- Answer questions about dishes\n"
            "- Suggest options based on dietary needs\n\n"
            "Try asking about vegetarian options, price ranges, or specific restaurants!"
        )
    
    if "help" in query_lower or "what can you" in query_lower:
        return (
            "Here's what I can help with:\n\n"
            "🍽️ **Menu Questions**\n"
            "- 'Show vegetarian options under ₹200'\n"
            "- 'What desserts does Restaurant X offer?'\n\n"
            "🏷️ **Price Info**\n"
            "- 'Most expensive dishes at Restaurant Y'\n"
            "- 'Cheapest North Indian food'\n\n"
            "🔍 **Comparisons**\n"
            "- 'Compare Restaurant A and B'\n"
            "- 'Which has better vegetarian options?'\n\n"
            "Just ask naturally like you would a knowledgeable friend!"
        )
    
    if "thank" in query_lower:
        return random.choice([
            "You're welcome! Happy to help.",
            "My pleasure! Let me know if you have other questions.",
            "Glad I could assist! 🍽️"
        ])
    
    if "how are you" in query_lower:
        return random.choice([
            "I'm doing well, thanks for asking! Ready to help with restaurant info.",
            "Doing great! Excited to help you find delicious options.",
            "I'm just a bot, but I'm happy to help with food questions!"
        ])
    
    return None

def handle_fallback(query: str) -> str:
    """Handle unrecognized queries"""
    return random.choice([
        "I'm sorry, I couldn't understand your question about restaurants. Could you please rephrase or specify?",
        "I specialize in restaurant information. Could you clarify your question about menus or dining options?",
        "I didn't quite catch that. Ask me about food, restaurants, or menus and I'll do my best to help!"
    ])


def handle_vegetarian_query(query: str) -> str:
    """Handle queries about vegetarian options"""
  
    veg_stats = []
    for restaurant in all_restaurants:
        restaurant_data = df[df['Name'] == restaurant]
        if not restaurant_data.empty:
            total_items = len(restaurant_data)
            veg_items = restaurant_data['is_vegetarian'].sum()
            veg_percentage = (veg_items / total_items) * 100
            veg_stats.append({
                'name': restaurant,
                'veg_count': veg_items,
                'total_items': total_items,
                'percentage': veg_percentage
            })
    
    if not veg_stats:
        return "I couldn't find any vegetarian options in our database."
    
  
    veg_stats.sort(key=lambda x: (-x['percentage'], -x['veg_count']))
    
  
    response = [
        "Here are restaurants with the best vegetarian options:\n",
        f"1. {veg_stats[0]['name']} ({veg_stats[0]['veg_count']} vegetarian items",
        f"2. {veg_stats[1]['name']} ({veg_stats[1]['veg_count']} vegetarian items",
        f"3. {veg_stats[2]['name']} ({veg_stats[2]['veg_count']} vegetarian items",
        "\nYou can ask about specific vegetarian dishes at any of these restaurants."
    ]
    
    return "\n".join(response)





def handle_comparison(restaurants: List[str], compare_by: str = "all") -> str:
    """
    Enhanced comparison with more metrics and filtering options
    
    Parameters:
    - restaurants: List of restaurant names to compare
    - compare_by: Specifies what to compare - 'all', 'cuisine', 'category', or a specific category/cuisine name
    
    Returns:
    - A formatted markdown comparison report
    """
    if len(restaurants) < 2:
        return "Please specify at least two restaurants to compare."
    
  
    if len(restaurants) > 2:
        restaurants = restaurants[:2]
        
    comparison_data = []
    for restaurant in restaurants:
        restaurant_data = df[df['Name'] == restaurant]
        if not restaurant_data.empty:
            
            cuisines = restaurant_data['Cuisines'].dropna().unique().tolist()
            categories = restaurant_data['Category'].dropna().unique().tolist()
            
            stats = {
                "name": restaurant,
                "avg_price": restaurant_data['clean_price'].mean(),
                "item_count": len(restaurant_data),
                "veg_percentage": (restaurant_data['is_vegetarian'].sum() / len(restaurant_data)) * 100,
                "categories": restaurant_data['Category'].value_counts().to_dict(),
                "cuisines": restaurant_data['Cuisines'].value_counts().to_dict(),
                "price_range": (restaurant_data['clean_price'].min(), restaurant_data['clean_price'].max()),
                "all_cuisines": cuisines,
                "all_categories": categories
            }
            comparison_data.append(stats)
    
    if len(comparison_data) < 2:
        return "Couldn't find data for both of the specified restaurants."
    
    
    report = [f"## {comparison_data[0]['name']} vs {comparison_data[1]['name']} Comparison\n"]
    
    
    if compare_by.lower() == 'cuisine':
        return generate_cuisine_comparison(comparison_data)
    
   
    elif compare_by.lower() == 'category':
        return generate_category_comparison(comparison_data)
    
   
    elif compare_by.lower() in [c.lower() for data in comparison_data for c in data["all_cuisines"]]:
        return generate_specific_cuisine_comparison(comparison_data, compare_by)
    
   
    elif compare_by.lower() in [c.lower() for data in comparison_data for c in data["all_categories"]]:
        return generate_specific_category_comparison(comparison_data, compare_by)
    
   
    else:
        # Price comparison
        report.append("### Price Comparison")
        for data in sorted(comparison_data, key=lambda x: x["avg_price"]):
            report.append(
                f"- **{data['name']}**: "
                f"Average ₹{data['avg_price']:.0f} "
                f"(Range: ₹{data['price_range'][0]} - ₹{data['price_range'][1]})"
            )
        
        # Vegetarian options
        report.append("\n### Vegetarian Options")
        for data in sorted(comparison_data, key=lambda x: x["veg_percentage"], reverse=True):
            report.append(
                f"- **{data['name']}**: "
                f"{data['veg_percentage']:.1f}% vegetarian ({int(data['item_count'] * data['veg_percentage']/100)} items)"
            )
        
        # Menu diversity
        report.append("\n### Top Menu Categories")
        for data in comparison_data:
            top_cats = ", ".join([f"{k} ({v})" for k, v in sorted(data["categories"].items(), key=lambda x: x[1], reverse=True)[:3]])
            report.append(f"- **{data['name']}**: {top_cats}")
            
        # Cuisine comparison
        report.append("\n### Cuisine Offerings")
        for data in comparison_data:
            top_cuisines = ", ".join([f"{k} ({v})" for k, v in sorted(data["cuisines"].items(), key=lambda x: x[1], reverse=True)[:3]])
            report.append(f"- **{data['name']}**: {top_cuisines}")
    
    return "\n".join(report)

def generate_cuisine_comparison(comparison_data: List[dict]) -> str:
    """Generate a cuisine-specific comparison between restaurants"""
    report = [f"## Cuisine Comparison: {comparison_data[0]['name']} vs {comparison_data[1]['name']}\n"]
    
   
    restaurant1_cuisines = set(comparison_data[0]["cuisines"].keys())
    restaurant2_cuisines = set(comparison_data[1]["cuisines"].keys())
    common_cuisines = restaurant1_cuisines.intersection(restaurant2_cuisines)
    
    
    if common_cuisines:
        report.append("### Common Cuisines")
        for cuisine in common_cuisines:
            r1_count = comparison_data[0]["cuisines"].get(cuisine, 0)
            r2_count = comparison_data[1]["cuisines"].get(cuisine, 0)
            report.append(f"- **{cuisine}**: {comparison_data[0]['name']}: {r1_count} items, {comparison_data[1]['name']}: {r2_count} items")
    
    # Unique cuisines
    report.append("\n### Unique Cuisines")
    
    # Restaurant 1 unique cuisines
    r1_unique = restaurant1_cuisines - restaurant2_cuisines
    if r1_unique:
        report.append(f"\n**Only in {comparison_data[0]['name']}**")
        for cuisine in r1_unique:
            report.append(f"- {cuisine}: {comparison_data[0]['cuisines'].get(cuisine, 0)} items")
    
    # Restaurant 2 unique cuisines
    r2_unique = restaurant2_cuisines - restaurant1_cuisines
    if r2_unique:
        report.append(f"\n**Only in {comparison_data[1]['name']}**")
        for cuisine in r2_unique:
            report.append(f"- {cuisine}: {comparison_data[1]['cuisines'].get(cuisine, 0)} items")
    
    return "\n".join(report)

def generate_category_comparison(comparison_data: List[dict]) -> str:
    """Generate a category-specific comparison between restaurants"""
    report = [f"## Menu Category Comparison: {comparison_data[0]['name']} vs {comparison_data[1]['name']}\n"]
    
   
    restaurant1_categories = set(comparison_data[0]["categories"].keys())
    restaurant2_categories = set(comparison_data[1]["categories"].keys())
    common_categories = restaurant1_categories.intersection(restaurant2_categories)
    
   
    if common_categories:
        report.append("### Common Categories")
        for category in common_categories:
            r1_count = comparison_data[0]["categories"].get(category, 0)
            r2_count = comparison_data[1]["categories"].get(category, 0)
            report.append(f"- **{category}**: {comparison_data[0]['name']}: {r1_count} items, {comparison_data[1]['name']}: {r2_count} items")
    
    # Unique categories
    report.append("\n### Unique Categories")
    
    # Restaurant 1 unique categories
    r1_unique = restaurant1_categories - restaurant2_categories
    if r1_unique:
        report.append(f"\n**Only in {comparison_data[0]['name']}**")
        for category in r1_unique:
            report.append(f"- {category}: {comparison_data[0]['categories'].get(category, 0)} items")
    
    # Restaurant 2 unique categories
    r2_unique = restaurant2_categories - restaurant1_categories
    if r2_unique:
        report.append(f"\n**Only in {comparison_data[1]['name']}**")
        for category in r2_unique:
            report.append(f"- {category}: {comparison_data[1]['categories'].get(category, 0)} items")
    
    return "\n".join(report)

def generate_specific_cuisine_comparison(comparison_data: List[dict], cuisine: str) -> str:
    """Generate a comparison for a specific cuisine between restaurants"""
    
    cuisine_proper = None
    for data in comparison_data:
        for c in data["all_cuisines"]:
            if c.lower() == cuisine.lower():
                cuisine_proper = c
                break
        if cuisine_proper:
            break
    
    if not cuisine_proper:
        return f"Cuisine '{cuisine}' not found in either restaurant."
    
    report = [f"## {cuisine_proper} Cuisine Comparison: {comparison_data[0]['name']} vs {comparison_data[1]['name']}\n"]
    

    for data in comparison_data:
        restaurant_data = df[(df['Name'] == data['name']) & (df['Cuisine'] == cuisine_proper)]
        
        if len(restaurant_data) == 0:
            report.append(f"**{data['name']}**: No {cuisine_proper} dishes available")
            continue
            
        report.append(f"### {data['name']} ({cuisine_proper})")
        
        # Item count
        report.append(f"- **Total {cuisine_proper} items**: {len(restaurant_data)}")
        
       
        avg_price = restaurant_data['clean_price'].mean()
        price_range = (restaurant_data['clean_price'].min(), restaurant_data['clean_price'].max())
        report.append(f"- **Average price**: ₹{avg_price:.0f} (Range: ₹{price_range[0]} - ₹{price_range[1]})")
        
        # Vegetarian percentage
        veg_percent = (restaurant_data['is_vegetarian'].sum() / len(restaurant_data)) * 100
        report.append(f"- **Vegetarian options**: {veg_percent:.1f}% ({int(len(restaurant_data) * veg_percent/100)} items)")
        
      
        top_categories = restaurant_data['Category'].value_counts().nlargest(3).to_dict()
        cat_list = ", ".join([f"{k} ({v})" for k, v in top_categories.items()])
        report.append(f"- **Top categories**: {cat_list}")
        
        report.append("")  
    
    return "\n".join(report)

def generate_specific_category_comparison(comparison_data: List[dict], category: str) -> str:
    """Generate a comparison for a specific category between restaurants"""
   
    category_proper = None
    for data in comparison_data:
        for c in data["all_categories"]:
            if c.lower() == category.lower():
                category_proper = c
                break
        if category_proper:
            break
    
    if not category_proper:
        return f"Category '{category}' not found in either restaurant."
    
    report = [f"## {category_proper} Category Comparison: {comparison_data[0]['name']} vs {comparison_data[1]['name']}\n"]
    
   
    for data in comparison_data:
        restaurant_data = df[(df['Name'] == data['name']) & (df['Category'] == category_proper)]
        
        if len(restaurant_data) == 0:
            report.append(f"**{data['name']}**: No {category_proper} items available")
            continue
            
        report.append(f"### {data['name']} ({category_proper})")
        
       
        report.append(f"- **Total {category_proper} items**: {len(restaurant_data)}")
        
       
        avg_price = restaurant_data['clean_price'].mean()
        price_range = (restaurant_data['clean_price'].min(), restaurant_data['clean_price'].max())
        report.append(f"- **Average price**: ₹{avg_price:.0f} (Range: ₹{price_range[0]} - ₹{price_range[1]})")
        
        # Vegetarian percentage
        veg_percent = (restaurant_data['is_vegetarian'].sum() / len(restaurant_data)) * 100
        report.append(f"- **Vegetarian options**: {veg_percent:.1f}% ({int(len(restaurant_data) * veg_percent/100)} items)")
        
        
        cuisines = restaurant_data['Cuisine'].value_counts().nlargest(3).to_dict()
        cuisine_list = ", ".join([f"{k} ({v})" for k, v in cuisines.items()])
        report.append(f"- **Top cuisines**: {cuisine_list}")
        
        report.append("")  
    
    return "\n".join(report)


def handle_budget_query(query: str) -> str:
    """Special handler for budget-related queries"""
    entities = extract_entities(query)
    budget = entities.get("price_range", 200)  
    
    
    dietary_filter = {}
    if entities.get("dietary"):
        dietary_filter["dietary"] = entities["dietary"]
    
   
    if entities.get("restaurants"):
        restaurant_filter = {"restaurants": entities["restaurants"]}
    else:
        restaurant_filter = {}
    
    
    cuisine_filter = {}
    if entities.get("cuisines"):
        cuisine_filter["cuisines"] = entities["cuisines"]
    
   
    combined_filter = {**{"price_range": budget}, **dietary_filter, **restaurant_filter, **cuisine_filter}
    budget_items = filter_dataframe(df, combined_filter)
    
    if budget_items.empty:
        return f"I couldn't find any items matching your budget criteria. Try a higher budget or different filters."
    
 
    budget_items = budget_items.sort_values('clean_price')
    
   
    restaurants = budget_items['Name'].unique()
    
    response = [f"Here are budget-friendly options under ₹{budget}:"]
    
    for restaurant in restaurants[:3]:  
        rest_items = budget_items[budget_items['Name'] == restaurant].head(2)
        response.append(f"\n**{restaurant}**:")
        for _, item in rest_items.iterrows():
            response.append(f"• {item['Item Name']} - ₹{item['clean_price']} ({item['Category']})")
    
    total_count = len(budget_items)
    if total_count > 6: 
        response.append(f"\nShowing 6 of {total_count} total budget-friendly options.")
    
    return "\n".join(response)

def handle_dish_query(query: str, dish_name: str) -> str:
    """Handle queries about specific dishes"""
   
    dish_info = df[df['Item Name'].str.contains(dish_name, case=False, na=False)]
    
    if dish_info.empty:
        return f"I couldn't find information about '{dish_name}'. Could you check the spelling or try another dish?"
    
    
    dish = dish_info.iloc[0]
    
   
    details = [f"**{dish['Item Name']}** at {dish['Name']}"]
    details.append(f"• **Price**: ₹{dish['clean_price']}")
    details.append(f"• **Category**: {dish['Category']}")
    details.append(f"• **Type**: {'Vegetarian' if dish['is_vegetarian'] else 'Non-Vegetarian'}")
    
    if dish['Description']:
        details.append(f"• **Description**: {dish['Description']}")
    
    if dish['Cuisines']:
        details.append(f"• **Cuisines**: {dish['Cuisines']}")
    
    
    similar_dishes = df[
        (df['Category'] == dish['Category']) & 
        (df['Name'] == dish['Name']) & 
        (df['Item Name'] != dish['Item Name'])
    ].sample(min(2, len(df)))
    
    if not similar_dishes.empty:
        details.append("\n**You might also like:**")
        for _, similar in similar_dishes.iterrows():
            details.append(f"• {similar['Item Name']} (₹{similar['clean_price']})")
    
    return "\n".join(details)



def handle_menu_query(query: str) -> str:
    """Handle specific menu-related queries"""
    
    entities = extract_entities(query)
    restaurants = entities["restaurants"]
    
    
    if not restaurants:
        return "I can provide menu information for our 8 restaurants. Could you specify which restaurant you're interested in?"
    
    
    restaurant = restaurants[0] 
    menu_items = df[df['Name'] == restaurant]
    
    if menu_items.empty:
        return f"I couldn't find menu information for {restaurant}. Please check the restaurant name and try again."
    
    
    if "popular" in query.lower() or "best" in query.lower():
        
        popular_items = menu_items.sort_values('clean_price', ascending=False).head(3)
        response = [f"Popular items at {restaurant}:"]
        for _, item in popular_items.iterrows():
            response.append(f"• **{item['Item Name']}** - ₹{item['clean_price']} ({item['Category']})")
        return "\n".join(response)
    
   
    categories = menu_items['Category'].value_counts().to_dict()
    response = [f"**{restaurant} Menu Information:**"]
    response.append(f"Total items: {len(menu_items)}")
    response.append("\nCategories:")
    for category, count in categories.items():
        response.append(f"• {category}: {count} items")
    response.append("\nAsk about specific categories or dishes for more details!")
    
    return "\n".join(response)

def handle_category_price_query(query: str) -> str:
    """Handle queries about price ranges for specific categories at restaurants."""
    
   
    query_lower = query.lower()
    
   
    matched_restaurants = [r for r in all_restaurants if r.lower() in query_lower]
    if not matched_restaurants:
        return "To tell you about price ranges, I need to know which restaurant you're asking about. Could you specify the restaurant name?"

    restaurant = matched_restaurants[0]

   
    category_terms = {
        "Dessert": ["dessert", "sweet", "ice cream", "cake"],
        "Main Course": ["main course", "main dish", "entree", "dinner"],
        "Breakfast": ["breakfast", "morning meal"],
        "Appetizer": ["appetizer", "starter", "starters"],
        "Beverage": ["beverage", "drinks", "juice", "coffee", "tea","Shakes"]
    }

    detected_categories = []
    for cat, keywords in category_terms.items():
        if any(k in query_lower for k in keywords):
            detected_categories.append(cat)
    
    if not detected_categories:
        return f"I can tell you about price ranges for different menu categories at {restaurant}. What category are you interested in (desserts, main course, etc.)?"

   
    restaurant_df = df[df['Name'].str.lower() == restaurant.lower()]
    if restaurant_df.empty:
        return f"I couldn't find any menu information for {restaurant}."

    
    restaurant_df['clean_price'] = pd.to_numeric(restaurant_df['clean_price'], errors='coerce')

    responses = []
    for cat in detected_categories:
        matches = restaurant_df[restaurant_df['Category'].str.lower().str.contains(cat.lower())]

        if matches.empty:
            responses.append(f"I couldn't find any items in the {cat} category at {restaurant}.")
            continue

        min_price = int(matches['clean_price'].min())
        max_price = int(matches['clean_price'].max())
        avg_price = matches['clean_price'].mean()
        item_count = len(matches)

        cheapest_item = matches.loc[matches['clean_price'].idxmin()]
        priciest_item = matches.loc[matches['clean_price'].idxmax()]

        category_response = [
            f"**{cat}** at **{restaurant}** ({item_count} items):",
            f"• Price range: ₹{min_price} to ₹{max_price}",
            f"• Average price: ₹{avg_price:.0f}",
            "Sample items:",
            f"• Cheapest: {cheapest_item['Item Name']} (₹{cheapest_item['clean_price']})",
            f"• Most expensive: {priciest_item['Item Name']} (₹{priciest_item['clean_price']})"
        ]

        responses.append("\n".join(category_response))

    return "\n\n".join(responses)

def handle_contact_query(restaurant_name: str) -> str:
    """Handle requests for restaurant contact information"""
   
    normalized_name = restaurant_name.lower().strip()
    
    
    matched_restaurant = None
    for name in RESTAURANT_CONTACTS:
        if normalized_name in name.lower():
            matched_restaurant = name
            break
    
    if not matched_restaurant:
        return f" Couldn't find contact information for '{restaurant_name}'. Please check the spelling or ask about another restaurant."
    
    contact_info = RESTAURANT_CONTACTS[matched_restaurant]
    
    response = [
        f"📞 **Contact Information for {matched_restaurant}**:",
        f"📍 Address: {contact_info['address']}",
        f"📱 Phone: {contact_info['phone']}",
        f"🍽️ Cuisine: {contact_info['cuisine']}"
    ]
    
  
    if contact_info['phone']:
        response.append(f"\n💡 Tip: Call {contact_info['phone']} for reservations or inquiries")
    
    return "\n".join(response)

RESTAURANT_CONTACTS = {
    "Hotel Prakash & Restaurant": {
        "address": "19, Civil lines, Roorkee",
        "phone": "+917895885082",
        "cuisine": "North Indian, Chinese"
    },
    "Punjabi Dhaba - P.D": {
        "address": "48, Civil Lines, Opposite Bus Stand, Roorkee",
        "phone": "+919219440520",
        "cuisine": "North Indian, Afghan, Chinese"
    },
    "Baap Of Rolls": {
        "address": "22, Civil Lines, Haridwar Road, Roorkee",
        "phone": "+911141182222",
        "cuisine": "Unknown"
    },
    "Foodbay": {
        "address": "Civil Lines, Roorkee",
        "phone": "+919720223366",
        "cuisine": "Unknown"
    },
    "The Cook House": {
        "address": "232/22, Civil Lines, Roorkee",
        "phone": "+918171264730",
        "cuisine": "Unknown"
    },
    "Hungry Point": {
        "address": "Malviya Chowk, Roorkee",
        "phone": "+919854540007",
        "cuisine": "Unknown"
    },
    "Tanisha's Restaurant Royal Hyderabadi Biryani": {
        "address": "29, Near Axess Bank, Roorkee",
        "phone": "+919520100700",
        "cuisine": "Hyderabadi, Biryani"
    },
    "Bhalla Vaishav Dhaba": {
        "address": "Purana Haridwar Road, Roorkee",
        "phone": "+919149290739",
        "cuisine": "North Indian"
    }
}




INTENT_LABELS = [
    "greeting",
    "restaurant_comparison",
    "find_cheapest_item",
    "find_most_expensive_item",
    "vegetarian_options",
    "menu_inquiry",
    "dish_details",
    "price_query",
    "budget_query",
    "general_question",
    "fallback",
    "restaurant_list"
    "category_query"
]


def process_query(query: str) -> str:
    """Main query processing pipeline using intent classification"""
    query_lower = query.lower()
    contact_phrases = [
        "contact info",
        "phone number",
        "address of",
        "how to contact",
        "reach",
        "call"
    ]
    if any(phrase in query_lower for phrase in contact_phrases):
       
        entities = extract_entities(query)
        if entities["restaurants"]:
            return handle_contact_query(entities["restaurants"][0])
        else:
            return "Please specify which restaurant's contact information you need (e.g., 'What is Hotel Prakash's phone number?')"

    
    intent_result = intent_classifier(query, INTENT_LABELS)
    predicted_intent = intent_result["labels"][0]

    
    entities = extract_entities(query)
    
    
    if ("price range" in query_lower or "how much" in query_lower or "what's the price" in query_lower) and \
       any(category in query_lower for category in ["dessert", "main course", "starter", "appetizer", "breakfast", "beverage", "menu"]):
        return handle_category_price_query(query)
    if predicted_intent=="budget_query":
        return handle_budget_query(query)  
    
    
    if predicted_intent=="greeting":
        return handle_greeting(query)
    if predicted_intent == "restaurant_comparison" and len(entities["restaurants"]) >= 2:
        return handle_comparison(entities["restaurants"])
        
    if predicted_intent == "find_cheapest_item":
        return handle_price_query(query, "cheapest")
        
    if predicted_intent == "find_most_expensive_item":
        return handle_price_query(query, "expensive")
        
    if predicted_intent == "vegetarian_options":
        return handle_vegetarian_query(query)
        
    if predicted_intent == "menu_inquiry":
        return handle_menu_query(query)
        
    if predicted_intent == "dish_details" and "Item Name" in entities:
        return handle_dish_query(query, entities["Item Name"])

    if predicted_intent == "restaurant_list":
        if len(all_restaurants) > 0:
            restaurant_list = "\n".join(f"• {restaurant}" for restaurant in sorted(all_restaurants))
            return f"Here are all the restaurants in our database:\n\n{restaurant_list}\n\nTotal: {len(all_restaurants)} restaurants"
        return "I couldn't find any restaurants in the database."    
    

        if any(word in query.lower() for word in ["budget", "cheap", "affordable", "low price", "inexpensive"]):
            return handle_budget_query(query)   

    filtered_data = filter_dataframe(df, entities)
   
    retrieved_chunks = retrieve_chunks(query)
    

    

    context_parts = []
    if retrieved_chunks:
        context_parts.append("Relevant information:\n" + "\n\n".join(retrieved_chunks))

    if not filtered_data.empty:
        sample_size = min(5, len(filtered_data))
        sample_items = filtered_data.sample(sample_size)
        context_parts.append("Relevant menu items:\n" + "\n".join(
            f"- {row['Item Name']} at {row['Name']} (₹{row['clean_price']}) - {row['Category']} - " + 
            f"{'Vegetarian' if row['is_vegetarian'] else 'Non-Vegetarian'}" +
            (f" - {row['Description']}" if row['Description'] else "")
            for _, row in sample_items.iterrows()
        ))

    if not context_parts:
        return handle_fallback(query)
    
    context = "\n\n".join(context_parts)

    return generate_response(
        query=query,
        context=context,
        conversation_history=list(st.session_state.conversation)
    )



df, text_chunks, all_restaurants, all_categories, all_cuisine_types = load_data()
embed_model = load_embedding_model()
embeddings = embed_model.encode(text_chunks, show_progress_bar=False)
index = build_faiss_index(embeddings.astype("float32"))
generator = load_generator()


with st.sidebar:
    st.title("🍽️ Restaurant Explorer")
    st.markdown("""
    **Ask about:**
    - Contacts & Address
    - Menu items & prices
    - Dietary options (vegetarian, gluten-free)
    - Restaurant comparisons
    - Price ranges
    """)
    
    st.markdown("---")
    st.subheader("Sample Questions")
    samples = [
        "Hi! What can you do?",
        "How to contact Hotel Prakash?",
        "Compare Restaurant Prakash and Restaurant Hungry Point",
        "List all the restaurant",
        "Give me address of Bhalla Vaishav Dhaba"
        
    ]
    for sample in samples:
        if st.button(sample, key=f"sample_{sample[:20]}"):
            st.session_state.user_input = sample
            st.session_state.run_query = True
    
    st.markdown("---")
    st.subheader("Data Overview")
    st.write(f"📊 **Restaurants:** {len(all_restaurants)}")
    st.write(f"🍲 **Menu Items:** {len(df)}")
    st.write(f"🏷️ **Categories:** {len(all_categories)}")
    st.write(f"🌍 **Cuisines:** {len(all_cuisine_types)}")





st.title("🤖 ZOTO - Restaurants AI Assistant")
st.caption("Ask me anything about contacts,menus, prices, dietary options, and more! of Roorkee Restaurants")



for role, message in st.session_state.conversation:
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message">👤 <strong>You:</strong> {message}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message bot-message">🤖 <strong>Assistant:</strong> {message}</div>',
            unsafe_allow_html=True
        )

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question:",
        key="input_text", 
        placeholder="E.g. 'Show vegetarian options under ₹200 at Hotel Prakash'"
    )
    submit_button = st.form_submit_button("Ask")

    if submit_button and user_input.strip():
        
        st.session_state.user_input = user_input
        st.session_state.run_query = True
        


if 'run_query' in st.session_state and st.session_state.run_query:
    user_query = st.session_state.user_input
    
    if user_query.strip():
        with st.spinner("Searching for answers..."):
            
            response = process_query(user_query)
            
           
            st.session_state.conversation.append(("user", user_query))
            st.session_state.conversation.append(("assistant", response))
        
        
        st.session_state.user_input = ""
        st.session_state.run_query = False
        
       
        st.rerun()



if st.session_state.conversation:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation.clear()
        st.rerun()

# Data explorer section
with st.expander("🔍 Explore Restaurant Data"):
    st.subheader("Filter Menu Items")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_restaurant = st.selectbox(
            "Restaurant",
            ["All"] + all_restaurants,
            key="filter_restaurant"
        )
        selected_category = st.selectbox(
            "Category",
            ["All"] + all_categories,
            key="filter_category"
        )
    with col2:
        price_range = st.slider(
            "Max Price (₹)",
            min_value=0,
            max_value=int(df['clean_price'].max()) + 100,
            value=int(df['clean_price'].max()) + 100,
            step=50,
            key="filter_price"
        )
        diet_filter = st.radio(
            "Diet",
            ["All", "Vegetarian", "Non-Vegetarian", "Gluten-Free"],
            key="filter_diet"
        )
    
   
    filtered = df.copy()
    if selected_restaurant != "All":
        filtered = filtered[filtered['Name'] == selected_restaurant]
    if selected_category != "All":
        filtered = filtered[filtered['Category'] == selected_category]
    if price_range > 0:
        filtered = filtered[filtered['clean_price'] <= price_range]
    if diet_filter == "Vegetarian":
        filtered = filtered[filtered['is_vegetarian'] == True]
    elif diet_filter == "Non-Vegetarian":
        filtered = filtered[filtered['is_vegetarian'] == False]
    elif diet_filter == "Gluten-Free":
        filtered = filtered[filtered['Description'].str.contains('gluten-free', case=False, na=False)]
    
   
    if not filtered.empty:
        st.write(f"Found {len(filtered)} items:")
        st.dataframe(
            filtered[['Name', 'Item Name', 'clean_price', 'Category', 'Vegetarian']],
            column_config={
                "Name": "Restaurant",
                "Item Name": "Dish",
                "clean_price": st.column_config.NumberColumn("Price (₹)"),
                "Vegetarian": "Veg"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No items match your filters.")