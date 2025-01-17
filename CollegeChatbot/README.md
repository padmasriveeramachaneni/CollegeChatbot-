# 🎓 College & Student Marks Chatbot

An AI-powered chatbot and student marks analyzer built using **Streamlit**, **Google Generative AI**, and **FAISS**. This application allows users to interact with a college information chatbot or analyze student marks using roll numbers.

---

## Features

### **1. College Info Chatbot**
- Provides detailed responses to queries about college programs, staff, and accreditations.
- Extracts knowledge from PDF documents using **PyPDF2** and stores it in a **FAISS** index for fast retrieval.
- Powered by **Google Generative AI** for natural language understanding and response generation.

### **2. Student Marks Analysis**
- Supports batch-wise and semester-wise result analysis for students.
- Validates roll numbers against predefined formats using regex.
- Reads and processes student data from CSV files to generate accurate responses.

### **3. User-Friendly Interface**
- Built with **Streamlit** for an intuitive and interactive user experience.
- Sidebar navigation for switching between "College Info" and "Student Marks" modes.

---

## Technologies Used

### **Backend**
- **Python**: Core programming language.
- **Google Generative AI (PaLM 2)**: For chatbot conversations and embeddings.
- **LangChain**: Manages conversational workflows.
- **FAISS**: Efficient similarity search and text storage.

### **Frontend**
- **Streamlit**: Web application framework for the user interface.

### **File Handling**
- **PyPDF2**: Extracts text from PDFs for chatbot knowledge base.
- **CSV Module**: Reads and processes student data.

### **Others**
- **Regex**: Validates roll numbers and processes input queries.

---

## Installation and Setup

### **1. Prerequisites**
- Python 3.8+
- Google Cloud API Key (for Generative AI integration)

### **2. Clone the Repository**
```bash
git clone https://github.com/yourusername/college-student-chatbot.git
cd college-student-chatbot

3. Install Dependencies
Create a virtual environment and install required packages:
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
pip install -r requirements.txt

4. Set Up API Key
Add your Google API key to the environment variables or replace the placeholder in the code:
GOOGLE_API_KEY = "your-google-api-key"

5. Run the Application
Start the Streamlit app:


