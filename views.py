from django.http import HttpResponse
from django.shortcuts import render
from BitcoinFuturePricePrediction import *

from utils import get_plot,get_train_test_original,get_original_future


def home(request):
	future_dates,output,live_price,df1,trainPredictPlot,testPredictPlot,dataStartDate,dataEndDate,Future_pred,Future_output_original = prediction()
	chart = get_train_test_original(df1,trainPredictPlot,testPredictPlot,dataStartDate,dataEndDate)
	return render(request,"home.html", {'chart':chart})


def result(request):
	future_dates,output,live_price,df1,trainPredictPlot,testPredictPlot,dataStartDate,dataEndDate,Future_pred,Future_output_original = prediction()
	chart = get_original_future(df1,Future_pred,Future_output_original)
	return render(request,"result.html",{'chart':chart})

def predicted(request):
	future_dates,output,live_price,df1,trainPredictPlot,testPredictPlot,dataStartDate,dataEndDate,Future_pred,Future_output_original = prediction()
	chart = get_plot(future_dates,output,live_price)
	return render(request,"predicted.html",{'chart':chart})
