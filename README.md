# Social Media Comment Analyzer

This project is a web-based application developed to analyze user comments on social media platforms. It performs sentiment analysis, spam detection, and toxicity checks on comments retrieved from platforms like YouTube and Instagram. Built with Python and Django, it features real-time data fetching, comment classification, and an interactive dashboard.

##  Project Description

The Social Media Comment Analyzer helps users understand the nature of comments on posts by classifying them into positive, negative, neutral, spam, and toxic. It uses machine learning models and Natural Language Processing (NLP) techniques to process the data and visualize the results in an easy-to-understand format.
Users can log in, paste a post URL, and instantly get an analysis report. The system is built for educational and research purposes, highlighting the power of sentiment analysis in today's digital communication.

##  Key Features

- User login and secure authentication
- Real-time fetching of comments using APIs
- Sentiment analysis (positive, neutral, negative)
- Detection of spam and toxic comments
- Categorized result display on a clean dashboard
- Easy-to-use form for URL submission
- Filtering options for toxic/spam comments

##  Key Skills Demonstrated

- Django web development (MVC pattern)
- Frontend styling with HTML, CSS, JS
- Integration of NLP libraries (NLTK, SpaCy)
- Machine Learning (classification models)
- YouTube/Instagram API integration (YouTube implemented)
- Model training and result display
- User session handling
- Collaboration and modular coding

##  Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Django)
- **APIs**: YouTube Data API v3 (Instagram in future)
- **NLP Libraries**: NLTK, SpaCy, TensorFlow
- **Database**: SQLite (default Django DB)
- **Authentication**: Django’s built-in user management

## Folder Structure

socialmedia_analyzer/
├── analyzer/
│ ├── templates/
│ │ ├── base.html
│ │ ├── login.html
│ │ ├── result.html
│ │ └── home.html
│ ├── views.py
│ ├── urls.py
│ └── models.py
├── media/
├── static/
│ ├── css/
│ └── js/
├── socialmedia_analyzer/
│ ├── settings.py
│ └── urls.py
├── db.sqlite3
├── manage.py
└── README.md


## How to Run the Project Locally

1. Clone or extract the repository.
2. Create a virtual environment and activate it:
3. Install dependencies:
4. Run migrations:
5. Start the server:
6. Open a browser and go to: `http://127.0.0.1:8000/`
7. Log in or register and start analyzing!

Note: You may need to add your YouTube API key in the `views.py` file or a `.env` config file.

##  Comment Categories

-  Positive
-  Neutral
-  Negative
-  Spam
-  Toxic

## Authors / Contributors

- **Pushpendra Singh Rajput** – Backend Developer , Frontend Developer , API Integrator & ML Model Trainer

