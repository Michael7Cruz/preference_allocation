from django.urls import path
from . import views

urlpatterns = [
    path('mainProg/', views.mainProg, name='mainProg'),
    #path('create/', views.create, name='create'),
    path('budgeting/', views.budgeting, name='budgeting'),
    path('', views.home, name='home'),
]