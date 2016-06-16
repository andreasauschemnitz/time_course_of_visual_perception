from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import Manager
import numpy as np
import matplotlib.pyplot as plt

nExe = 64

#case = "short-very"
case = "short"
#case = "long"

def callSimulate(idRunLocal,results):
	global idRun
	idRun = idRunLocal
	execfile("visionModel.py")
	results[idRun] = np.array(winner)
	#execfile("test.py")
	#results[idRun] = np.array(r)

manager = Manager()
results = manager.dict()

if __name__ == '__main__':
	pool = Pool(processes = 16) #use all available cores, otherwise specify the number you want as an argument
	for i in range(nExe):
		pool.apply_async(callSimulate, args=(i,results))
	pool.close()
	pool.join()
	
	'''	
	ps = [Process(target = callSimulate, args=(iExe,results)) for iExe in range(nExe)]
	
	for p in ps:
		p.start()
	
	for p in ps:
		p.join()
	'''

finalErr = 0
finalIC = 0
finalCorrect = 0
for i in range(len(results)):
	r=results[i]
	
	err = (r==0)
	ic = (r==1)
	correct = (r==2)
	
	finalErr += err
	finalIC += ic
	finalCorrect += correct

finalErr = 1.0 * finalErr / nExe
finalIC = 1.0 * finalIC / nExe
finalCorrect = 1.0 * finalCorrect / nExe

xticks=np.arange(50,801,50)

plt.plot(finalErr,label="error", c="blue")
plt.plot(finalIC,label="correct", c="red")
plt.plot(finalCorrect,label="IC", c="green")

color_stimulus_edge = "#ffa500"
color_stimulus_face = "#ffedcc"

if case == "short-very":
	tStimulusEnd = 200
elif case == "short":
	tStimulusEnd = 250
else:
	tStimulusEnd = 550

plt.axvspan(150, tStimulusEnd, facecolor=color_stimulus_face, edgecolor=color_stimulus_edge)

plt.xticks(xticks-50, xticks)
plt.xlim( (0,750) )
plt.ylim( (0,1) )
plt.legend()

plt.xlabel("time / ms")
plt.ylabel("probability")
plt.title("Decision of the network (mean over runs)")

directory = "out/"
plt.savefig(directory+'winner_total.png', bbox_inches='tight')
plt.clf()
plt.close()

print("all processes finished")

