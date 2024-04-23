from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = socForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_age=request.POST.get('Age')
            data_es=request.POST.get('EstimatedSalary')
            data_gm=request.POST.get('Gender_Male')
            try:
                data_age = float(data_age)
                data_es = float(data_es)
                data_gm = float(data_gm)
            except ValueError:
                return redirect(self.failure_url)

# Create youimport pandas as pd

            dataset=pd.read_csv("Social_Network_Ads.csv")
            dicc={'yes':1,'no':0}
            dataset=pd.get_dummies(dataset,drop_first=True)
            dataset=dataset.drop("User ID",axis=1)
            independent=dataset[['Age','EstimatedSalary','Gender_Male']]
            dep=dataset["Purchased"]
            from sklearn.model_selection import train_test_split
            X_train,X_test,Y_train,Y_test=train_test_split(independent,dep,test_size=1/3,random_state=0)
            #predicting the output value using Gaussian Navie Bayes classifier
            from sklearn.naive_bayes import GaussianNB
            classifier=GaussianNB()
            classifier.fit(X_train,Y_train)
            Y_pred=classifier.predict(X_test)
            from sklearn.metrics import confusion_matrix
            cm=confusion_matrix(Y_test,Y_pred)
            from sklearn.metrics import classification_report
            clf_report=classification_report(Y_test,Y_pred)
            #print(clf_report)
            #print(cm)
            data = np.array([data_age,data_es,data_gm])
            #data = sc.fit_transform(data.reshape(-1,1))
            out=classifier.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=ckd()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')
            return render(request, "succ_msg.html", {'data_age':data_age,'data_es':data_es,'data_gm':data_gm,'out':out})


        else:
            return redirect(self.failure_url)
