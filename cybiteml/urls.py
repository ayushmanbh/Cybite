from django.urls import path
from django.conf.urls import url
from . import views
from .views import ProjectDetailView, UserProjectListView, ProjectCreateView, ProjectUpdateView, ProjectDeleteView


urlpatterns = [
    path('', views.home, name='home'),
    url(r'^output_auto', views.output_auto, name='script_auto'),
    path('user/<str:username>', UserProjectListView.as_view(), name='user_projects'),
    path('project/<int:pk>/', ProjectDetailView.as_view(), name='project_detail'),
    path('project/new/', ProjectCreateView.as_view(), name='project_create'),
    path('project/<int:pk>/update', ProjectUpdateView.as_view(), name='project_update'),
    path('project/<int:pk>/delete', ProjectDeleteView.as_view(), name='project_delete'),

]