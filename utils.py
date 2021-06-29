import matplotlib.pyplot as plt
import base64
from io import BytesIO
import seaborn as sns


def get_graph():
	buffer = BytesIO()
	plt.savefig(buffer,format='png')
	buffer.seek(0)
	image_png = buffer.getvalue()
	graph = base64.b64encode(image_png)
	graph = graph.decode('utf-8')
	buffer.close()
	return graph


def get_plot(future_dates,output,live_price):
	plt.switch_backend('AGG')

	sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
	sns.set_style( {'grid.linestyle': '--'})

	
	fig, axi = plt.subplots(figsize=(12,5))
	axi.xaxis.grid(False)
	axi.tick_params( colors='white')

	axi.plot_date(future_dates, output,'r', label = 'Predicted Future Price')
	  
	axi.plot(future_dates[0],live_price,'ro',color='g',label = 'Actual Price') 
	plt.legend(loc = 'upper right',labelcolor='linecolor')
	plt.xlabel("Days",color = 'white')
	plt.ylabel('Price(USD)',color='white')
	plt.title("Bitcoin Price Prediction since "+ future_dates[0],color='white')
	plt.xticks(rotation=90)
		
	plt.tight_layout()
	graph = get_graph()
	return graph

def get_train_test_original(df1,trainPredictPlot,testPredictPlot,dataStartDate,dataEndDate):
	plt.switch_backend('AGG')

	sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
	sns.set_style( {'grid.linestyle': '--'})

	ax = plt.axes()
	
	ax.tick_params( colors='white')


	plt.plot(df1,'b', label = "Original Price")
	plt.plot(trainPredictPlot, 'r', label = 'Training set')
	plt.plot(testPredictPlot,'g', label = 'Test set')

	plt.legend(loc = 'upper left',labelcolor='linecolor')
	plt.xlabel("Days",color = 'white')
	plt.ylabel('Price(USD)',color='white')
	plt.title("Bitcoin Price from "+ dataStartDate + "to " + dataEndDate,color='white')
	plt.tight_layout()
	graph = get_graph()

	return graph

def get_original_future(df1,Future_pred,Future_output_original):
	plt.switch_backend('AGG')

	sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
	sns.set_style( {'grid.linestyle': '--'})

	ax = plt.axes()
	
	ax.tick_params( colors='white')
	
	plt.plot(df1,'c', label = "Original Price")
	plt.plot(Future_pred,Future_output_original,'y', label = 'Future Predicted Price')
	plt.legend(loc = 'upper left',labelcolor='linecolor')
	plt.xlabel("Days",color = 'white')
	plt.ylabel('Price(USD)',color='white')
	plt.title("Bitcoin Price with future predictions",color='white')
	plt.tight_layout()

	graph = get_graph()
	return graph
