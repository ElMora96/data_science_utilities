import numpy as np
import pandas as pd
#Simple exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as SES

class ThetaMethod():
#Theta model for forecasting (Standard). Non-seasonal version.
#Using two theta lines.
#Following Dudek's guidelines for load forecasting. 
	def __init__(self, theta_1, theta_2):
		'''Initialize theta model

		Parameters.
		theta_1: float in [0,1)
		To describe long-term behavior of series.
		theta_2: float > 1
		To describe short-term behavior of series.
		'''
		self.theta_1 = theta_1
		self.theta_2 = theta_2

	def fit(self):
		'''Fit model to given time series. Compute fixed parameters a and b,
		which are independent from thetas.
		'''
		self.n = len(self.series)
		self.t_vec = pd.Series(np.arange(1, self.n + 1), index = self.series.index) #time vector

		#Compute parameter b
		b = 2*np.mean(self.t_vec*self.series) - (1 + self.n)*np.mean(self.series) #partial
		self.b = 6*(1/(self.n**2 -1))*b 

		#Compute parameter a 
		self.a = np.mean(self.series) - (self.n+1)*(1/2)*self.b


	def theta_line_forecast(self, theta):
		'''One step ahead prediction using theta line extrapolation
		'''
		#Compute theta lines
		if theta == 0:
			forecast = self.a + self.b*self.t_vec[-1]
			new_index = [self.series.index[-1] + pd.Timedelta(1,"H")]
			forecast = pd.Series(forecast, index = new_index)

		else:
			theta_line = theta*self.series + (1 - theta)*(self.a + self.b*self.t_vec)

			#Simple exponential smoothing model
			model = SES(theta_line).fit() #eventually add smoothing_level = ...

			#one-step ahead forecast
			forecast = model.forecast(1)

		return forecast

	def forecast(self, series, steps = 1):
		'''Produce recursive forecasts.
		Parameters
		series: pd.Series
		Time series to forecast using theta method.
		steps: int.
		Number of steps to forecast.
		'''
		#Initialize time series
		self.series = series

		for step in range(steps):
			#Fit model to series
			self.fit()

			#Make theta line extrapolation
			extrapolation_1 = self.theta_line_forecast(self.theta_1)
			extrapolation_2 = self.theta_line_forecast(self.theta_2)

			#Combine forecast - equal weights
			extrapolation = (1/2)*extrapolation_1 + (1/2)*extrapolation_2

			#Append value to series
			self.series = self.series.append(extrapolation)

		forecast = self.series[-steps:]
		return forecast






		
		
