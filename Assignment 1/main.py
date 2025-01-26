import matplotlib.pyplot as plt
from data_generation import generateSamples
from threshold_classification import findOptimalThreshold
from likelihood_classification import classifyByLikelihood
from quantization_classification import classifyByQuantization

samplesCount = 1000
femaleMean = 152
maleMean = 166
sdInitial = 5
sdList = [2.5, 5, 7.5, 10]
quantIntervals = [0.001, 0.05, 0.1, 0.3, 1, 2, 5, 10]

for sd in sdList:
    maleSamples = generateSamples(maleMean, sd, samplesCount)
    femaleSamples = generateSamples(femaleMean, sd, samplesCount)

    print(f"\n-----------------------------------\nSD = {sd}\n-----------------------------------\n")

    # Threshold Classification
    bestThreshold, thresholdError, thresholdCM = findOptimalThreshold(maleSamples, femaleSamples, samplesCount)
    print(f"Optimal Threshold: {bestThreshold}\nThreshold Error: {thresholdError:.2f}%\nConfusion Matrix: {thresholdCM}\n")

    # Likelihood Classification
    likelihood_error, likelihood_cm = classifyByLikelihood(maleSamples, femaleSamples, samplesCount)
    print(f"Likelihood Classification Error: {likelihood_error:.2f}%\nConfusion Matrix: {likelihood_cm}\n")

    # Quantization Classification
    for interval in quantIntervals:
        quant_error, quant_cm = classifyByQuantization(maleSamples, femaleSamples, interval, samplesCount)
        print(f"Quantization Interval = {interval}, Quantization Error: {quant_error:.2f}%, Confusion Matrix: {quant_cm}")

plt.hist(maleSamples, bins=50, alpha=0.5, label='Male Samples', color='blue')
plt.hist(femaleSamples, bins=50, alpha=0.5, label='Female Samples', color='pink')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Height Distributions for Male and Female')
plt.show()