from django.urls import path
from . import views

urlpatterns = [
    path('', views.input_form, name='input_form'),
    path('summarize/', views.summarize_text, name='summarize_text'),
    path('translator/', views.translator_form, name='translator_form'),
    path('paraphrase/', views.paraphraser_form, name='paraphraser_form'),
    path('translate/', views.translate_text, name='translate_text'),  # Translation endpoint
    path('paraphrase/', views.paraphrase_text, name='paraphrase_text'),  # Paraphrase endpoint
]

