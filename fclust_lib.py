#Class to perform fuzzy clustering
#Libraries
import numpy as np
from itertools import combinations

class FuzzyCMeans():
	"""docstring for Fcm"""
	def __init__(self, X, n_clusters = 3, m = 1.5):
		'''
		X: data matrix
		m: fuzziness parameter
		'''
		self.X = X
		self.T = X.shape[1] #TS length
		self.n_units = len(self.X)
		self.n_clusters = n_clusters
		self.m = m
		self.Membership_Matrix = self.Initialize_Membership_Matrix()
		self.Centroid_Matrix = self.Compute_Centroids()

	def dist(self,x,y):
		#Distance function to be used during clustering - Euclidean
		return np.linalg.norm(x - y)

	def Initialize_Membership_Matrix(self):    
		#Uniform initial assigment
		#A random one; other zeros
		Membership_Matrix = np.zeros(shape = (self.n_units, self.n_clusters))   
		for j in range(self.n_units):
			rand = np.random.randint(0, self.n_clusters)
			Membership_Matrix[j, rand] = 1
		return Membership_Matrix

	def New_Centroid(self, c):
		#c: int. Cluster label
		memberships = self.Membership_Matrix[:, c]
		summation_vec = np.power(memberships, np.repeat(self.m, self.n_units))
		summation = summation_vec.sum()
		weighted_sum = np.zeros(self.T)
		for j in range(self.n_units):
			newval = X[j, :] * summation_vec[j]
			weighted_sum += newval
		centroid = weighted_sum/summation
		return centroid

	def Compute_Centroids(self):
		'''
		Returns
		-------
		Centroid_Matrix : np matrix
			Matrix with TS of centroids on rows
		'''    
		Centroid_Matrix = np.empty((self.n_clusters, self.T))
		for c in range(self.n_clusters):
			centroid = self.New_Centroid(c)
			Centroid_Matrix[c,:] = centroid
		
		return Centroid_Matrix

	def Membership_Value(self, ts, c):              
		'''
		memb : float in [0,1]
			Membership to cluster c.
		'''
		centroid_list = self.Centroid_Matrix.tolist()
		distance_t_c = self.dist(ts, centroid_list[c])
		summation_vec = [distance_t_c/self.dist(ts, ctr) for ctr in centroid_list]
		summation_vec = np.power(summation_vec, np.repeat((2/(self.m-1)), self.n_clusters))
		summation = summation_vec.sum()
		memb = np.power(summation, -1)
		return memb

	def Update_Membership_Matrix(self):

		New_Membership_Matrix = np.empty((self.n_units, self.n_clusters))
		for i in range(self.n_units):
			for j in range(self.n_clusters):
				New_Membership_Matrix[i,j] = self.Membership_Value(X[i,:], j)
		return New_Membership_Matrix

	#Cluster Validity Criteria
	def Partition_Coefficient(self):
		temp_matrix = np.square(self.Membership_Matrix)
		PC = temp_matrix.sum()/n_units
		return PC

	def Partition_Entropy(self):
		transform = lambda x : x*np.log(x)
		v_transform = np.vectorize(transform) #vectorize lambda 
		temp_matrix = v_transform(self.Membership_Matrix)
		PE = - (1/float(n_units)) * temp_matrix.sum()
		return PE

	def Modified_Partition_Coefficient(self):
		PC = Partition_Coefficient()
		MPC = 1 - (self.n_clusters/float(self.n_clusters-1))*(1-PC)
		return MPC

	def Xie_Beni(self):
		summation_matrix = np.empty((self.n_units, self.n_clusters))
		for i in range(self.n_units):
			for j in range(self.n_clusters):
				summation_matrix[i,j]  = np.power(self.Membership_Matrix[i,j], self.m)*np.power(self.dist(self.X[i,:], self.Centroid_Matrix[j,:]),2)
		summation = summation_matrix.sum()
		pair_list = combinations(list(self.Centroid_Matrix), 2)
		distance_list = [self.dist(a,b) for(a, b) in pair_list]
		min_dist = min(distance_list)
		XB = summation/(self.n_units*min_dist)
		return XB

	def Cluster_Defuzzification(self):
		df = pd.DataFrame(self.Membership_Matrix)
		crisp_clusters = df.idxmax(axis = 1) #Series with defuzzified clusters
		return crisp_clusters

	#Fuzzy Silhouette
	def S_i(self, i):
		#Silhouette score for single observation
		#i : time series index
		ts = self.X[i]
		cluster_i = self.crisp_clusters[i]
		other_clusters = [lab for lab in range(self.n_clusters) if lab != cluster_i]
		same_cluster_indices = self.crisp_clusters[self.crisp_clusters == cluster_i].index
		clusters_counts = self.crisp_clusters.value_counts(sort = False)
		#Measure of compactness
		a_i = 0
		for j in same_cluster_indices:
			D = np.power(self.dist(ts, self.X[j]), 2)
			a_i += D
		a_i = a_i/clusters_counts[cluster_i]
		
		#Measure of separation
		d_i = [] #Average distances from all other clusters
		for clust in other_clusters: 
			val = 0
			for k in self.crisp_clusters[self.crisp_clusters == clust].index:
				D_k = np.power(self.dist(ts, self.X[k]), 2)
				val += D_k
			val = val/clusters_counts[clust]
			d_i.append(val)       
		b_i = min(d_i)
		
		#Silhouette
		S = (b_i - a_i)/max(a_i, b_i)
		return S

	def Fuzzy_Silhouette(self, gamma = 0.5):
		#Gamma is a weighting coefficient
		summation_vec = np.empty(self.n_units)
		silhouette_vec = np.empty(self.n_units) #store silhouettes
		self.crisp_clusters = self.Cluster_Defuzzification() #Defuzzy 
		for i in range(self.n_units):
			tmp = self.Membership_Matrix[i, :]
			tmp.sort()
			summation_vec[i] = tmp[-1] - tmp[-2] #1st & 2nd largest element difference
			silhouette_vec[i] = self.S_i(i)
		summation_vec = np.power(summation_vec, np.repeat(gamma, self.n_units))
		num_vec = np.multiply(summation_vec, silhouette_vec)
		num = num_vec.sum()
		den = summation_vec.sum()
		return num/den

	#Procedure to fit model
	def fit(self, iter = 10):
		for j in range(iter):
			self.Centroid_Matrix = self.Compute_Centroids()
			self.Membership_Matrix = self.Update_Membership_Matrix()


