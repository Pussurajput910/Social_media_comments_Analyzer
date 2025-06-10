# views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.conf import settings

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import urllib

from emoji import demojize
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import re

from transformers import pipeline, AutoTokenizer

# =============== Configuration =================
YOUTUBE_API_KEY = ('AIzaSyCxRPz0VTqOedR_iCiq8c1n_G7zXKDcbC0')  # Safer way to access API key
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
classifier = pipeline('sentiment-analysis')
# Toxic and Spam classifiers
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
spam_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# =============== Emoji Sentiment Mapping ============
EMOJI_SENTIMENT = {
    'orange_heart': 'Positive',    # 🧡
    'yellow_heart': 'Positive',    # 💛
    'green_heart': 'Positive',     # 💚
    'blue_heart': 'Positive',      # 💙
    'purple_heart': 'Positive',    # 💜
    'brown_heart': 'Positive',     # 🤎
    'black_heart': 'Negative',     # 🖤
    'gray_heart': 'Positive',      # 🩶
    'white_heart': 'Positive',     # 🤍
    'pink_heart': 'Positive',      # 🩷
    'heart_with_arrow': 'Positive', # 💘
    'heart_with_ribbon': 'Positive', # 💝
    'sparkling_heart': 'Positive',  # 💖
    'growing_heart': 'Positive',    # 💗
    'beating_heart': 'Positive',    # 💓
    'revolving_hearts': 'Positive', # 💞
    'grinning_face': 'Positive',    # 😀
    'grinning_face_with_big_eyes': 'Positive', # 😃
    'grinning_face_with_smiling_eyes': 'Positive', # 😄
    'beaming_face_with_smiling_eyes': 'Positive',  # 😁
    'grinning_squinting_face': 'Positive', # 😆
    'smiling_face_with_sweat': 'Positive', # 😅
    'face_with_tears_of_joy': 'Positive',  # 😂
    'rolling_on_the_floor_laughing': 'Positive', # 🤣
    'winking_face': 'Positive',    # 😉
    'kissing_face': 'Positive',    # 😗
    'kissing_face_with_smiling_eyes': 'Positive', # 😙
    'kissing_face_with_closed_eyes': 'Positive',  # 😚
    'face_blowing_a_kiss': 'Positive',  # 😘
    'smiling_face_with_hearts': 'Positive', # 🥰
    'heart_eyes': 'Positive',      # 😍
    'star_struck': 'Positive',     # 🤩
    'partying_face': 'Positive',   # 🥳
    'smiling_face': 'Positive',    # 😊
    'slightly_smiling_face': 'Positive', # 🙂
    'relieved_face': 'Positive',   # 😌
    'smiling_face_with_tear': 'Positive', # 🥲
    'face_with_pleading_eyes': 'Positive', # 🥹
    'upside_down_face': 'Neutral', # 🙃
    'man_and_woman_holding_hands': 'Neutral', # 👫
    'man_and_woman_walking': 'Neutral', # 🚶‍♂️🚶‍♀️
    'smirking_face': 'Neutral',    # 😏
    'drooling_face': 'Neutral',    # 🤤
    'face_savoring_food': 'Neutral', # 😋
    'face_with_tongue': 'Neutral', # 😛
    'zany_face': 'Neutral',        # 🤪
    'woozy_face': 'Negative',      # 🥴
    'pensive_face': 'Negative',    # 😔
    'pleading_face': 'Negative',   # 🥺
    'grimacing_face': 'Negative',  # 😬
    'expressionless_face': 'Negative', # 😑
    'neutral_face': 'Negative',    # 😐
    'face_without_mouth': 'Negative', # 😶
    'face_in_clouds': 'Negative',  # 😶‍🌫
    'dotted_line_face': 'Negative', # 🫥
    'zipper_mouth_face': 'Negative', # 🤐
    'saluting_face': 'Negative',   # 🫡
    'thinking_face': 'Negative',   # 🤔
    'shushing_face': 'Negative',   # 🤫
    'face_with_open_eyes_and_hand_over_mouth': 'Negative', # 🫢
    'face_with_monocle': 'Negative', # 🧐
    'yawning_face': 'Negative',    # 🥱
    'hugging_face': 'Negative',    # 🤗
    'face_with_raised_eyebrow': 'Negative', # 🤨
    'face_screaming_in_fear': 'Negative', # 😱
    'face_with_thermometer': 'Negative', # 🤒
    'unamused_face': 'Negative',   # 😒
    'face_with_rolling_eyes': 'Negative', # 🙄
    'face_exhaling': 'Negative',   # 😮‍💨
    'angry_face': 'Negative',      # 😤
    'pouting_face': 'Negative',    # 😠
    'enraged_face': 'Negative',    # 😡
    'face_with_symbols_on_mouth': 'Negative', # 🤬
    'disappointed_face': 'Negative', # 😞
    'downcast_face_with_sweat': 'Negative', # 😓
    'worried_face': 'Negative',    # 😟
    'disappointed_relieved_face': 'Negative', # 😥
    'crying_face': 'Negative',     # 😢
    'frowning_face': 'Negative',   # ☹
    'slightly_frowning_face': 'Negative', # 🙁
    'confused_face': 'Negative',   # 😕
    'anxious_face_with_sweat': 'Negative', # 😰
    'fearful_face': 'Negative',    # 😨
    'anguished_face': 'Negative',  # 😧
    'frowning_face_with_open_mouth': 'Negative', # 😦
    'astonished_face': 'Negative', # 😮
    'hushed_face': 'Negative',     # 😯
    'face_with_open_mouth': 'Negative', # 😲
    'flushed_face': 'Negative',    # 😳
    'exploding_head': 'Negative',  # 🤯
    'confounded_face': 'Negative', # 😖
    'persevering_face': 'Negative', # 😣
    'weary_face': 'Negative',      # 😩
    'dizzy_face': 'Negative',      # 😵
    'dizzy_face_with_spiral_eyes': 'Negative', # 😵‍💫
    'face_with_diagonal_mouth': 'Negative', # 🫨
    'cold_face': 'Negative',       # 🥶
    'hot_face': 'Negative',        # 🥵
    'nauseated_face': 'Negative',  # 🤢
    'face_vomiting': 'Negative',   # 🤮
    'sleeping_face': 'Negative',   # 😴
    'sleepy_face': 'Negative',     # 😪
    'sneezing_face': 'Negative',   # 🤧
    'head_bandage': 'Negative',    # 🤕
    'face_with_medical_mask': 'Negative', # 😷
    'lying_face': 'Negative',      # 🤥
    'ogre': 'Negative',            # 👹
    'goblin': 'Negative',          # 👺
    'skull': 'Negative',           # 💀
    'angry_face_with_horns': 'Negative', # 👿
}

# =============== Helper Functions ===================

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith(('/embed/', '/v/')):
                return parsed_url.path.split('/')[2]
    except Exception as e:
        print(f"[extract_video_id] Error: {e}")
    return None

def fetch_youtube_comments(video_id):
    comments = []
    next_page_token = None
    try:
        while True:
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response.get('items', []):
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                replies = item.get('replies', {}).get('comments', [])
                comments.extend(reply['snippet']['textDisplay'] for reply in replies)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    except Exception as e:
        print(f"[fetch_youtube_comments] Error: {e}")

    return comments

def analyze_comment(comment):
    try:
        # Sentiment Analysis
        comment_text = demojize(comment)
        emoji_matches = re.findall(r':(.*?):', comment_text)
        emoji_sentiments = [EMOJI_SENTIMENT.get(emoji.lower()) for emoji in emoji_matches if emoji.lower() in EMOJI_SENTIMENT]

        tokens = tokenizer.tokenize(comment)
        if len(tokens) > 512:
            comment = tokenizer.convert_tokens_to_string(tokens[:512])

        text_analysis = classifier(comment)[0]
        text_sentiment = text_analysis['label'].capitalize()
        if text_analysis['score'] > 0.6:
            emoji_sentiments.append('Positive' if text_sentiment == 'Positive' else 'Negative')
        sentiment = max(set(emoji_sentiments), key=emoji_sentiments.count) if emoji_sentiments else 'Neutral'

        # Toxicity Detection
        toxic_result = toxicity_classifier(comment)[0]
        is_toxic = toxic_result['label'].lower() == 'toxic' and toxic_result['score'] > 0.6

        # Spam Detection
        spam_result = spam_classifier(comment)[0]
        is_spam = spam_result['label'].lower() == 'spam' and spam_result['score'] > 0.6

        return {
            'text': comment,
            'sentiment': sentiment,
            'is_toxic': is_toxic,
            'is_spam': is_spam
        }

    except Exception as e:
        print(f"[analyze_comment] Error: {e}")
        return {'text': comment, 'sentiment': 'Neutral', 'is_toxic': False, 'is_spam': False}

def analyze_comments(comments):
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    spam_count = 0
    toxic_count = 0
    analyzed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(analyze_comment, comments)

    for result in results:
        sentiment = result['sentiment'].lower()
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        if result['is_spam']:
            spam_count += 1
        if result['is_toxic']:
            toxic_count += 1
        analyzed.append(result)

    return sentiment_counts, spam_count, toxic_count, analyzed

def generate_pie_chart(sentiment_data):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiment_data['positive'], sentiment_data['neutral'], sentiment_data['negative']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"

# =============== Django Views ===================

def index(request):
    return render(request, 'analyzer/index.html')

def login_view(request):
    if request.method == 'POST':
        user = authenticate(
            request, 
            username=request.POST['username'], 
            password=request.POST['password']
        )
        if user:
            login(request, user)
            return redirect('analyze')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'analyzer/login.html')

def signup_view(request):
    if request.method == 'POST':
        username, password, email = request.POST['username'], request.POST['password'], request.POST['email']
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
        else:
            User.objects.create_user(username=username, password=password, email=email)
            messages.success(request, 'Account created successfully.')
            return redirect('login')
    return render(request, 'analyzer/signup.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def analyze_view(request):
    if request.method == 'POST':
        url = request.POST['url']
        video_id = extract_video_id(url)
        if not video_id:
            messages.error(request, 'Invalid YouTube URL.')
            return redirect('analyze')

        comments = fetch_youtube_comments(video_id)
        if not comments:
            messages.error(request, 'No comments found or unable to fetch comments.')
            return redirect('analyze')

        sentiment_results, spam_count, toxic_count, analyzed_comments = analyze_comments(comments)
        request.session['sentiment_results'] = json.dumps(sentiment_results)
        request.session['analyzed_comments'] = json.dumps(analyzed_comments)
        request.session['spam_count'] = spam_count
        request.session['toxic_count'] = toxic_count

        return redirect('result')

    return render(request, 'analyzer/analyze.html')

def result_view(request):
    sentiment_results = request.session.get('sentiment_results')
    analyzed_comments = request.session.get('analyzed_comments')

    if not (sentiment_results and analyzed_comments):
        messages.error(request, 'No analysis data found.')
        return redirect('analyze')

    sentiment_results = json.loads(sentiment_results)
    analyzed_comments = json.loads(analyzed_comments)

    # Calculate the counts for positive, neutral, and negative sentiments
    positive_count = sentiment_results.get('positive', 0)
    neutral_count = sentiment_results.get('neutral', 0)
    negative_count = sentiment_results.get('negative', 0)
    spam_count = request.session.get('spam_count', 0)
    toxic_count = request.session.get('toxic_count', 0)
    # Generate the pie chart
    pie_chart = generate_pie_chart(sentiment_results)

    # Prepare the context to be passed to the template
    context = {
    'total_comments': len(analyzed_comments),
    'positive_count': sentiment_results.get('positive', 0),
    'neutral_count': sentiment_results.get('neutral', 0),
    'negative_count': sentiment_results.get('negative', 0),
    'spam_count': spam_count,
    'toxic_count': toxic_count,
    'comments': analyzed_comments,
    'pie_chart': pie_chart
}

    # Render the result template with the context
    return render(request, 'analyzer/result.html', context)
