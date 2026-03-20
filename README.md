# 🎬 Movie Recommendation System

A complete **content-based Movie Recommendation System** built with **Python**, **scikit-learn**, and **Streamlit**.

This project uses two datasets:
- `titles.csv`
- `credits.csv`

The system merges both datasets, performs feature engineering on metadata such as **overview**, **genres**, **cast**, and **director**, then recommends the **top 5 similar movies** using **CountVectorizer** and **cosine similarity**.

It also includes a **Netflix-style Streamlit web app** with a dark theme, searchable movie selection, loading spinner, sidebar details, and a clean five-column recommendation layout.

---

## 📌 Project Objective

The goal of this project is to build a **content-based recommendation engine** without using any external movie API.

Instead of fetching posters or metadata from online services, the application works entirely from local CSV files and generates recommendations based on movie content and credits.

This is useful for:
- Machine Learning / Data Science assignments
- portfolio projects
- recommendation system demos
- understanding feature engineering with text data

---

## 🚀 Features

### Backend / ML Features
- Load `titles.csv` and `credits.csv` using **pandas**
- Merge datasets on the `id` column
- Support multiple common schema formats:
  - TMDB-style datasets with JSON-like `genres`, `cast`, and `crew`
  - row-based credits datasets with `name`, `character`, and `role`
- Select and process relevant metadata:
  - `title`
  - `overview`
  - `genres`
  - `cast`
  - `crew`
- Drop rows with missing required values
- Extract:
  - top 3 cast members
  - director from crew
- Convert movie metadata into a single `tags` feature
- Normalize tokens by:
  - converting to lowercase
  - removing spaces from names
- Vectorize text with **CountVectorizer(max_features=5000, stop_words='english')**
- Compute similarity with **cosine_similarity**
- Return top 5 recommended movies

### Streamlit UI Features
- Dark **Netflix-style** theme
- Main title: **🎬 Movie Recommendation System**
- Subtitle: **Get similar movies instantly**
- Searchable movie dropdown
- Manual text input option
- `Recommend` button
- Loading spinner for better UX
- 5-column recommendation layout using `st.columns`
- Placeholder poster image for all recommendations
- Sidebar with:
  - About Project
  - Tech Stack
- Footer: **Built with ❤️ using Machine Learning**

---

## 🗂️ Project Structure

```bash
Data_Science-AIML_Projects/
│
├── app.py                  # Streamlit web app
├── movie_recommender.py    # Python recommender module / CLI entry point
├── README.md               # Project documentation
├── titles.csv              # Input dataset (user-provided)
└── credits.csv             # Input dataset (user-provided)
```

---

## ⚙️ How the Recommendation System Works

### 1. Data Loading
Both CSV files are loaded using pandas.

### 2. Data Normalization and Merge
The project normalizes title and credits schemas and merges both datasets on `id`.

### 3. Feature Engineering
The app extracts useful metadata such as:
- overview words
- genre names
- top 3 cast members
- director name

These are combined into a single text column called **`tags`**.

### 4. Text Vectorization
The `tags` column is transformed into numerical vectors using:
- `CountVectorizer(max_features=5000, stop_words='english')`

### 5. Similarity Computation
Cosine similarity is calculated between all movie vectors.

### 6. Recommendation
When a user selects a movie, the model finds the 5 most similar movies and displays them in the UI.

---

## 🧠 Tech Stack

- **Python**
- **Pandas**
- **scikit-learn**
- **Streamlit**
- **AST (`ast.literal_eval`)**

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Data_Science-AIML_Projects
```

### 2. Create and activate a virtual environment
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install pandas scikit-learn streamlit
```

---

## 📁 Dataset Requirements

Place the following files in the project root directory:
- `titles.csv`
- `credits.csv`

### Expected Title Data
The titles dataset should contain:
- `id`
- `title` or `name`
- `overview` or `description`
- `genres` or `genre`

### Expected Credits Data
The credits dataset should contain either:

#### Option A: TMDB-style
- `id`
- `cast`
- `crew`

#### Option B: Row-based credits
- `id`
- `name`
- `character` *(optional but helpful)*
- `role` *(optional but helpful)*

---

## ▶️ Run the Python Recommender Script

To run the standalone recommender module:

```bash
python movie_recommender.py
```

This will attempt to build the recommender and print the top 5 similar movies for a sample title.

---

## 🌐 Run the Streamlit Web App

Start the app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

---

## 💡 How to Use the Web App

1. Launch the app with `streamlit run app.py`
2. Select a movie from the dropdown **or** type a movie name manually
3. Click the **Recommend** button
4. View 5 recommended movies in card layout

---

## ❗ Troubleshooting

### Error: `titles.csv or credits.csv was not found`
Make sure both dataset files are placed in the root project folder.

### Error: `Movie not found`
Try selecting a movie directly from the dropdown list.

### Error related to missing columns
Check whether your CSV files contain the expected columns listed above.

### Streamlit not found
Install Streamlit:
```bash
pip install streamlit
```

---

## 📈 Future Improvements

Possible next enhancements:
- add real local poster assets instead of a generic placeholder
- add fuzzy movie search suggestions
- add genre filters
- add top-rated / popular sections
- add model persistence with pickle
- add unit tests for preprocessing and recommendation logic
- deploy the app on Streamlit Community Cloud or similar platform

---

## ❤️ Footer Note

Built with ❤️ using Machine Learning.
