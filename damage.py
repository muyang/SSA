from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd

import georasters as gr
import matplotlib.pyplot as plt
# Load data
r1 = 'D:/gis/soil.asc'
r2 = 'D:/gis/dist.asc'
r3 = 'D:/gis/fcov.asc'

depth = gr.from_file(r1)
landuse = gr.from_file(r2)
pop = gr.from_file(r3)

depth = depth*3

#calc damage value, or function of model
'''
def damage(landuse,param):
	depth = param[0]
	weight = param[1]
'''	
def damage(landuse,depth,weight):
	if landuse < 0.2:
		if depth < 1:
			return 0.1*depth*weight
		elif depth < 2:
			return (0.2*(depth-1)+0.2)*weight
		elif depth < 3:
			return (0.1*(depth-2)+0.3)*weight
		else:
			return 0.4*weight
	elif landuse < 0.4:
		if depth < 1:
			return 0.15*depth*weight
		elif depth < 2:
			return (0.2*(depth-1)+0.15)*weight
		elif depth < 3:
			return (0.15*(depth-2)+0.35)*weight
		else:
			return 0.5*weight
	elif landuse < 0.8:
		if depth < 1:
			return 0.12*depth*weight
		elif depth < 2:
			return (0.25*(depth-1)+0.12)*weight
		elif depth < 3:
			return (0.1*(depth-2)+0.37)*weight
		else:
			return 0.47*weight
	else: 
		if depth < 2:
			return 0.15*depth*weight
		elif depth < 3:
			return (0.25*(depth-2)+0.3)*weight
		else:
			return 0.55*weight
			

landuse = landuse.to_pandas()
weight = pop.to_pandas()
depth = depth.to_pandas()

S=[]
SS=[]

for k in range(1369):
	d=depth.value[k]
	w=weight.value[k]
	problem = {
		'num_vars': 2,
		'names': ['depth', 'weight'],
		'bounds': [[0.9*d,1.2*d+0.001],     #'bounds':[a,b], a<b
				   [0.5*w, 1.4*w+0.001]]
	}
	param_values = saltelli.sample(problem, 100, calc_second_order=True) 
	#run model
	Y = np.empty([param_values.shape[0]])
	#Y = damage(landuse,param_values)
	for i, X in enumerate(param_values):
		Y[i] = damage(landuse.value[k],X[0],X[1])
	#print Y
	Si = sobol.analyze(problem, Y, print_to_console=False)
	#print Si['S1'] #,Si['ST']
	S.append(Si['S1'][0])
	SS.append(Si['S1'][1])
	
#df = pd.DataFrame(S)
#print df

df = pop.to_pandas()
df.value=S
raster0 = gr.from_pandas(df, value='value',x='x',y='y')
#f1=plt.subplot(121)
#plt.sca(f1)
raster0.plot()

df.value=SS
raster1 = gr.from_pandas(df, value='value',x='x',y='y')
#f2=plt.subplot(122)
#plt.sca(f2)
#plt.plot(raster1)
raster1.plot()

plt.show()

'''
raster = from_pandas(df, value='value', x='x', y='y', cellx= cellx, celly=celly, xmin=xmin, ymax=ymax,
					GeoT=GeoT, nodata_value=NDV, projection=Projection, datatype=DataType)
'''