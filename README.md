# 🍽️ ZOTO – Restaurant AI Chatbot for Roorkee

**ZOTO** is a smart Streamlit-based conversational chatbot powered by Retrieval-Augmented Generation (RAG), built to explore menus, prices, dietary preferences, and more from restaurants in **Roorkee, India**.



## 🧠 Features

- 🔍 Query specific dishes, restaurants, and menus
- 💬 Chat with an intelligent assistant that understands:
  - Price queries
  - Dietary restrictions
  - Restaurant comparisons
  - Contact/address info
- 🤖 Uses **FAISS** + **MiniLM** for retrieval and **Flan-T5** for generation
- 📊 Data explorer to interactively filter menu items by restaurant, category, diet, and price
- 🌱 Fully local, no paid APIs used

---

## 📁 Project Structure

```
├── app.py                            # Main Streamlit chatbot app
├── zomato_restaurants_final_fixed.csv  # Cleaned dataset from Zomato
├── Restaurant_Data_Scrapping.ipynb   # Jupyter notebook for scraping restaurant data
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🧪 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/EMPER0R7/Restaurant_Assistant_ChatBot.git
cd Restaurant_Assistant_ChatBot
```

### 2. Create & activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser. If not, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

---

## 💡 Example Queries You Can Ask

- “Show vegetarian options under ₹200”
- “Compare Tamra and Tanisha’s Restaurant”
- “What are the desserts at Indian Accent?”
- “Does Hungry Point have any gluten-free dishes?”
- “Compare beverages at Tamra and Baap Of Rolls”

---

## ⚙️ Tech Stack

| Tool                  | Purpose                                |
|-----------------------|----------------------------------------|
| `Streamlit`           | Chatbot frontend interface             |
| `sentence-transformers` | For generating semantic embeddings    |
| `faiss-cpu`           | Fast retrieval of similar items        |
| `transformers`        | Language model (`Flan-T5`) for responses |
| `pandas`              | Data processing                        |
| `Jupyter`             | For scraping and preprocessing menus   |

---



## 🙌 Acknowledgements

- 🤗 [Hugging Face](https://huggingface.co) for open-source models
- 🍴 [Zomato](https://zomato.com) for menu and restaurant data


---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---


