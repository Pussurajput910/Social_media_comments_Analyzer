from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('analyze/', views.analyze_view, name='analyze'),
    path('result/', views.result_view, name='result'),
]
