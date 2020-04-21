from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from .models import Project
from .forms import DataForm
from django.contrib.auth.models import User
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
# from django.utils.inspect import get_func_full_args

# import inspect
from random import choice

from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tpot.builtins import StackingEstimator

global df

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

training_features, testing_features, training_target, testing_target = train_test_split(iris.data, iris.target,
                                                                                        random_state=None)

titanic = pd.read_csv('train.csv')
titanic.rename(columns={'Survived': 'target'}, inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})
titanic = titanic.fillna(-999)
titanic_new = titanic.drop(['Name', 'Ticket', 'Cabin', 'target'], axis=1)
features = titanic_new
training_features_x, testing_features_x, training_classes_x, testing_classes_x = train_test_split(features,
                                                                                                  titanic['target'],
                                                                                                  random_state=None)

X, y = pd.DataFrame(data=iris.data, columns=iris.feature_names), pd.DataFrame(data=iris.target, columns=["iris_type"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_train, y_test = np.ravel(y_train), np.ravel(y_test)


def on_iris():
    # Average CV score on the training set was: 0.9818181818181818
    exported_pipeline = make_pipeline(
        Normalizer(norm="l1"),
        StackingEstimator(estimator=LogisticRegression(C=0.0001, dual=False, penalty="l2")),
        DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=14, min_samples_split=6)
    )
    exported_pipeline.fit(training_features, training_target)
    return exported_pipeline


def on_titanic():
    exported_pipeline = RandomForestClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1,
                                               min_samples_split=9)
    exported_pipeline.fit(training_features_x, training_classes_x)
    return exported_pipeline


def output_auto(request):
    func_chosen = choice([on_iris, on_titanic])
    if func_chosen == on_iris:
        results = on_iris().predict(testing_features)
        df_name = 'Iris'
        html_table = df.head().to_html(index=False)
    else:
        results = on_titanic().predict(testing_features_x)
        df_name = 'Titanic'
        html_table = titanic_new.head().to_html(index=False)
    return render(request, 'cybiteml/home.html', {'results': results, 'html_table': html_table, 'df_name': df_name})


def home(request):
    if request.method == 'POST':
        form = DataForm(request.POST)
        if form.is_valid():
            size = form.cleaned_data['size']
            missing_q = form.cleaned_data['missing_q']
            missing_val = form.cleaned_data['missing_val']
            algo = form.cleaned_data['algo']
            if size == 'head':
                if missing_q == 'yes':
                    df.fillna(missing_val, inplace=True)
                    size_choice = df.head(10)
                else:
                    size_choice = df.head(10)
            else:
                if missing_q == 'yes':
                    df.fillna(missing_val, inplace=True)
                    size_choice = df
                else:
                    size_choice = df
            size_choice = size_choice.to_html(index=False)
            if algo == 'gnb':
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                return render(request, 'cybiteml/home.html', {'disp_df': size_choice,
                                                              'y_pred': y_pred, 'score': score})
            elif algo == 'lrc':
                logreg = LogisticRegression(random_state=0, solver="lbfgs",max_iter=400, multi_class="multinomial")
                logreg.fit(X_train, y_train)
                y_pred = logreg.predict(X_test)
                score = logreg.score(X_test, y_test)
                return render(request, 'cybiteml/home.html',
                              {'disp_df': size_choice, 'y_pred': y_pred, 'score': score})
            else:
                svc = SVC(kernel="linear", C=1.0, gamma="auto")
                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)
                score = svc.score(X_test, y_test)
                return render(request, 'cybiteml/home.html',
                              {'disp_df': size_choice, 'y_pred': y_pred, 'score': score})
    else:
        form = DataForm()
    return render(request, 'cybiteml/home.html', {'form': form})


class UserProjectListView(ListView):
    model = Project
    template_name = 'cybiteml/user_projects.html'
    context_object_name = 'projects'
    paginate_by = 2

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Project.objects.filter(creator=user).order_by('-date_created')


class ProjectDetailView(DetailView):
    model = Project


class ProjectCreateView(LoginRequiredMixin, CreateView):
    model = Project
    fields = ['title', 'description', 'file']

    def form_valid(self, form):
        form.instance.creator = self.request.user
        return super().form_valid(form)


class ProjectUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Project
    fields = ['title', 'description', 'file']

    def form_valid(self, form):
        form.instance.creator = self.request.user
        return super().form_valid(form)

    def test_func(self):
        project=self.get_object()
        if self.request.user == project.creator:
            return True
        else:
            return False


class ProjectDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Project
    success_url = '/'

    def test_func(self):
        project=self.get_object()
        if self.request.user == project.creator:
            return True
        else:
            return False



