import random

"""
# Physics constants
G = 9.8 
M_CART = 1.0
M_POLE = 0.1
L = 0.5
F = 100
DELTA_T = 0.02
PI = np.pi
MU_C = 0.0005
MU_P = 0.000002
"""

if __name__ == "__main__":
	random.seed(123456)
	params = {"G" : [8,12], "M_CART":[0.5,1.5],"M_POLE":[0.1,0.9],"L":[0.1,1], "F" : [5,20]}
	k_params = params.keys()
	#job_pattern = open("pendulum_exec_pattern", "r").read()
	for i in range(10):
		conf_file = open("CONFIG_PENDULUM_" + str(i), "w+")
		#job_file = open("JOB_PENDULUM_" + str(i), "w+")
		for key in k_params:
			interval = params[key]
			if len(interval) == 1:
				conf_file.write(key + "=" + str(interval[0]) + "\n")
			else:
				conf_file.write(key + "=" + str(random.uniform(interval[0],interval[1])) + "\n")
		conf_file.write("VIDEO_PREFIX=PENDULUM_VIDEO_"+str(i)+"_"+str(i) + "\n")
		conf_file.close()
		job_file.write(job_pattern.replace(":i:", str(i)))
		
		
