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

# ---------------------------
# Configuration
# ---------------------------
MAX_HISTORY_LENGTH = 10  # Keep last 10 messages in memory
DEFAULT_K = 5  # Number of chunks to retrieve
TEMPERATURE = 0.3  # Controls response creativity (0-1)

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title=" ZOMO Restaurant Chatbot",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left: 4px solid #4e79a7;
    }
    .user-message {
        background-color: #e7f5ff;
        border-left: 4px solid #228be6;
    }
    .restaurant-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
    }
    .highlight {
        background-color: #fff3bf;
        padding: 0 2px;
        border-radius: 3px;
    }
    .sidebar-content {
        padding: 10px;
    }
    .warning {
        color: #e67700;
        font-weight: 500;
    }
    .success {
        color: #2b8a3e;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data Loading
# ---------------------------
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
        
        # Enhanced text representation for better retrieval
        df['text'] = df.apply(lambda row: (
            f"Menu Item: {row['Item Name']}\n"
            f"Restaurant: {row['Name']}\n"
            f"Price: ₹{row['clean_price']}\n"
            f"Category: {row['Category']}\n"
            f"Description: {row['Description']}\n"
            f"Diet: {'Vegetarian' if row['Vegetarian'].strip().lower() == 'yes' else 'Non-Vegetarian'}\n"
            f"Cuisines: {row['Cuisines']}"
        ), axis=1)
        
        # Create structured fields for filtering
        df['is_vegetarian'] = df['Vegetarian'].str.strip().str.lower() == 'yes'
        df['cuisine_list'] = df['Cuisines'].str.split(',').apply(lambda x: [c.strip() for c in x] if isinstance(x, list) else [])
        
        # Get unique values for UI
        restaurants = sorted(df['Name'].unique())
        categories = sorted(df['Category'].unique())
        cuisine_types = sorted(set([c for sublist in df['cuisine_list'] for c in sublist if c]))
        
        return df, df['text'].tolist(), restaurants, categories, cuisine_types
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), [], [], [], []

# ---------------------------
# Model Loading
# ---------------------------
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

# ---------------------------
# Core Functions
# ---------------------------

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
    
    # Extract restaurants (more robust matching)
    for restaurant in all_restaurants:
        if re.search(r'\b' + re.escape(restaurant.lower()) + r'\b', query_lower):
            entities["restaurants"].append(restaurant)
    
    # Extract categories
    for category in all_categories:
        if re.search(r'\b' + re.escape(category.lower()) + r'\b', query_lower):
            entities["categories"].append(category)
    
    # Extract cuisines
    for cuisine in all_cuisine_types:
        if re.search(r'\b' + re.escape(cuisine.lower()) + r'\b', query_lower):
            entities["cuisines"].append(cuisine)
    
    # Price range extraction (more patterns)
    price_patterns = [
        r"(?:under|below|less than|cheaper than|up to|maximum|max)\s*₹?\s*(\d+)",
        r"₹?\s*(\d+)\s*(?:and under|or less|or below)",
        r"(\d+)\s*rupees?"
    ]
    for pattern in price_patterns:
        if match := re.search(pattern, query_lower):
            entities["price_range"] = int(match.group(1))
            break
    
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
    
    # Comparison detection
    entities["comparison"] = any(
        word in query_lower
        for word in ["compare", "versus", "vs", "difference between", "which is better"]
    ) and len(entities["restaurants"]) >= 2
    
    return entities

def filter_dataframe(df: pd.DataFrame, entities: Dict) -> pd.DataFrame:
    """Enhanced dataframe filtering with better cuisine handling"""
    filtered = df.copy()
    
    # Restaurant filter
    if entities["restaurants"]:
        filtered = filtered[filtered['Name'].isin(entities["restaurants"])]
    
    # Category filter
    if entities["categories"]:
        filtered = filtered[filtered['Category'].isin(entities["categories"])]
    
    # Cuisine filter (more accurate with pre-split cuisine_list)
    if entities["cuisines"]:
        cuisine_filter = filtered['cuisine_list'].apply(
            lambda x: any(c.lower() in [cl.lower() for cl in entities["cuisines"]] for c in x)
        )
        filtered = filtered[cuisine_filter]
    
    # Price filter
    if entities["price_range"] is not None:
        filtered = filtered[filtered['clean_price'] <= entities["price_range"]]
    
    # Dietary filter
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
            temperature=0.7,  # Slightly more creative
            num_beams=3
        )[0]['generated_text']
        
        # Add some variety to common responses
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

# ---------------------------
# Specialized Handlers
# ---------------------------
def handle_greeting(query: str) -> Optional[str]:
    """Handle greetings and introductory questions"""
    greeting_keywords = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
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
    
    return None

def handle_fallback(query: str) -> str:
    """Handle unrecognized queries"""
    return random.choice([
        "I'm sorry, I couldn't understand your question about restaurants. Could you please rephrase or specify?",
        "I specialize in restaurant information. Could you clarify your question about menus or dining options?",
        "I didn't quite catch that. Ask me about food, restaurants, or menus and I'll do my best to help!"
    ])
def handle_comparison(restaurants: List[str]) -> str:
    """Enhanced comparison with more metrics"""
    if len(restaurants) < 2:
        return "Please specify at least two restaurants to compare."
    
    comparison_data = []
    for restaurant in restaurants:
        restaurant_data = df[df['Name'] == restaurant]
        if not restaurant_data.empty:
            stats = {
                "name": restaurant,
                "avg_price": restaurant_data['clean_price'].mean(),
                "item_count": len(restaurant_data),
                "veg_percentage": (restaurant_data['is_vegetarian'].sum() / len(restaurant_data)) * 100,
                "categories": restaurant_data['Category'].value_counts().nlargest(3).to_dict(),
                "price_range": (restaurant_data['clean_price'].min(), restaurant_data['clean_price'].max())
            }
            comparison_data.append(stats)
    
    if not comparison_data:
        return "Couldn't find data for the specified restaurants."
    
    # Generate comparison report
    report = ["## Restaurant Comparison Report\n"]
    
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
        top_cats = ", ".join([f"{k} ({v})" for k, v in data["categories"].items()])
        report.append(f"- **{data['name']}**: {top_cats}")
    
    # Additional metrics could be added here (cuisine diversity, etc.)
    
    return "\n".join(report)

def handle_price_query(query: str, price_type: str = "cheapest") -> str:
    """Improved price query handling"""
    entities = extract_entities(query)
    filtered = filter_dataframe(df, entities)
    
    if filtered.empty:
        return "No matching items found. Try broadening your search."
    
    # Determine sort order
    ascending = price_type == "cheapest"
    sorted_df = filtered.sort_values('clean_price', ascending=ascending)
    
    # Get top 5 items
    top_items = sorted_df.head(5)
    
    # Generate response
    price_term = "cheapest" if ascending else "most expensive"
    response = [f"Here are the {price_term} matching items:"]
    
    for _, row in top_items.iterrows():
        item_info = [
            f"**{row['Item Name']}**",
            f"₹{row['clean_price']}",
            f"at {row['Name']}",
            f"({row['Category']})"
        ]
        if row['Description']:
            item_info.append(f"- {row['Description']}")
        response.append(" ".join(item_info))
    
    return "\n\n".join(response)

# ---------------------------
# Main Application
# ---------------------------
# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = deque(maxlen=MAX_HISTORY_LENGTH)

# Load data and models
df, text_chunks, all_restaurants, all_categories, all_cuisine_types = load_data()
embed_model = load_embedding_model()
embeddings = embed_model.encode(text_chunks, show_progress_bar=False)
index = build_faiss_index(embeddings.astype("float32"))
generator = load_generator()

# Sidebar
with st.sidebar:
    st.title("🍽️ Restaurant Explorer")
    st.markdown("""
    **Ask about:**
    - Menu items & prices
    - Dietary options (vegetarian, gluten-free)
    - Restaurant comparisons
    - Price ranges
    """)
    
    st.markdown("---")
    st.subheader("Sample Questions")
    samples = [
        "Show me vegetarian options under ₹200",
        "Compare Restaurant A and Restaurant B",
        "What's the most expensive dish at Restaurant C?",
        "Find gluten-free appetizers",
        "Which restaurant has the best variety of North Indian food?"
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

# Main interface
st.title("🍽️ Restaurant AI Assistant")
st.caption("Ask me anything about menus, prices, dietary options, and more!")

# Chat input
with st.form("chat_form"):
    user_input = st.text_input(
        "Your question:",
        value=getattr(st.session_state, "user_input", ""),
        key="user_input",
        placeholder="E.g. 'Show vegetarian options under ₹200'"
    )
    submit_button = st.form_submit_button("Ask")

# Process query
# if submit_button or getattr(st.session_state, "run_query", False):
#     if st.session_state.run_query:
#         user_input = st.session_state.user_input
#         st.session_state.run_query = False
    
#     if user_input.strip():
#         with st.spinner("Searching for answers..."):
#             # Add user message to history
#             st.session_state.conversation.append(("user", user_input))
#             if greeting_response := handle_greeting(user_input):
#                st.write(greeting_response)
#                st.stop()

#             # Extract entities for targeted retrieval
#             entities = extract_entities(user_input)
            
#             # Handle special query types
#             # if entities["comparison"]:
#             #     response = handle_comparison(entities["restaurants"])
#             # elif "cheapest" in user_input.lower():
#             #     response = handle_price_query(user_input, "cheapest")
#             # elif "most expensive" in user_input.lower():
#             #     response = handle_price_query(user_input, "expensive")
#             # else:
#             #     # Standard RAG flow
#             #     retrieved_chunks = retrieve_chunks(user_input)
#             #     filtered_data = filter_dataframe(df, entities)
            
#             if entities["comparison"]:
#                 return handle_comparison(entities["restaurants"])
    
#             if "cheapest" in query.lower():
#                 return handle_price_query(user_input, "cheapest")
    
#             if "most expensive" in query.lower():
#                 return handle_price_query(user_input, "expensive")
            
#     # Standard RAG flow
#             retrieved_chunks = retrieve_chunks(user_input)
#             filtered_data = filter_dataframe(df, entities)
            
#             # If we found nothing relevant
#             if not retrieved_chunks and filtered_data.empty:
#                 return handle_fallback(user_input)
                
#                 # Build context
#                 context_parts = []
#                 if retrieved_chunks:
#                     context_parts.append("Relevant menu information:\n" + "\n\n".join(retrieved_chunks))
                
#                 if not filtered_data.empty:
#                     sample_items = filtered_data.sample(min(3, len(filtered_data)))
#                     context_parts.append("Matched items:\n" + "\n".join(
#                         f"- {row['Item Name']} (₹{row['clean_price']}) at {row['Name']}"
#                         for _, row in sample_items.iterrows()
#                     ))
                
#                 context = "\n\n".join(context_parts) if context_parts else "No specific context found."
                
#                 # Generate response
#                 response = generate_response(
#                     query=user_input,
#                     context=context,
#                     conversation_history=list(st.session_state.conversation)
#                 )
            
#             # Add bot response to history
#             st.session_state.conversation.append(("assistant", response))

if submit_button or getattr(st.session_state, "run_query", False):
    if st.session_state.run_query:
        user_input = st.session_state.user_input
        st.session_state.run_query = False
    
    if user_input.strip():
        with st.spinner("Searching for answers..."):
            # Add user message to history
            st.session_state.conversation.append(("user", user_input))

            # Handle greeting
            if greeting_response := handle_greeting(user_input):
                st.write(greeting_response)
                st.stop()

            # Extract entities
            entities = extract_entities(user_input)

            # Handle special query types
            if entities.get("comparison"):
                response = handle_comparison(entities["restaurants"])
                st.write(response)
                st.stop()

            if "cheapest" in user_input.lower():
                response = handle_price_query(user_input, "cheapest")
                st.write(response)
                st.stop()

            if "most expensive" in user_input.lower():
                response = handle_price_query(user_input, "expensive")
                st.write(response)
                st.stop()

            # Standard RAG flow
            retrieved_chunks = retrieve_chunks(user_input)
            filtered_data = filter_dataframe(df, entities)

            if not retrieved_chunks and filtered_data.empty:
                response = handle_fallback(user_input)
                st.write(response)
                st.stop()

            # Build context
            context_parts = []

            if retrieved_chunks:
                context_parts.append("Relevant menu information:\n" + "\n\n".join(retrieved_chunks))

            if not filtered_data.empty:
                sample_items = filtered_data.sample(min(3, len(filtered_data)))
                context_parts.append("Matched items:\n" + "\n".join(
                    f"- {row['Item Name']} (₹{row['clean_price']}) at {row['Name']}"
                    for _, row in sample_items.iterrows()
                ))

            context = "\n\n".join(context_parts) if context_parts else "No specific context found."

            # Generate response
            response = generate_response(
                query=user_input,
                context=context,
                conversation_history=list(st.session_state.conversation)
            )

            # Add bot response to history
            st.session_state.conversation.append(("assistant", response))
            st.write(response)


# Display conversation
for i, (role, message) in enumerate(st.session_state.conversation):
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

# Clear conversation button
if st.session_state.conversation:
    if st.button("Clear Conversation"):
        st.session_state.conversation.clear()
        st.experimental_rerun()

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
    
    # Apply filters
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
    
    # Display results
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