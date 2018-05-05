import json
import numpy as np
from scipy import stats



with open('ttest.txt', 'r') as o:
	scoresMSE = json.load(o)
	keys = scoresMSE.keys()
	tTestStatistics = {}
	length = 0
	for i in range(len(keys)):
		for j in range(i + 1, len(keys)):
			iScores = scoresMSE[keys[i]]
			jScores = scoresMSE[keys[j]]
			length = len(iScores)
			sigma = np.sqrt(sum([(a - b) ** 2 for (a, b) in zip(iScores, jScores)]) / (len(iScores) - 1))
			tTestStatistics[keys[i] + " & " + keys[j]] = (np.mean(iScores) - np.mean(jScores)) / (sigma / np.sqrt(len(iScores)))

	with open('tTestStats.txt', 'w') as o:
		for key in scoresMSE:
			o.write(key + ' mean: ' + str(np.mean(scoresMSE[key])) + '\n')
	
		o.write('\n')

		for key in tTestStatistics:
			o.write(key + '\n')
			o.write(str(tTestStatistics[key]) + '\n')
			# two sided tests
			if tTestStatistics[key] < -stats.t.ppf(1-0.025, length) or tTestStatistics[key] > stats.t.ppf(1-0.025, length):
				o.write("We reject the null; these come from different distributions" + '\n')

			else:
				o.write("We do not reject the null; these may come from the same distribution" + '\n')