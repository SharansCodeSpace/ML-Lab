import numpy as np

def classifyByQuantization(maleSamples, femaleSamples, interval_length, sample_count):
    bins = np.arange(min(femaleSamples), max(maleSamples), interval_length)
    maleHist, _ = np.histogram(maleSamples, bins)
    femaleHist, _ = np.histogram(femaleSamples, bins)

    tp = fn = fp = tn = 0

    for height in maleSamples:
        binIndex = np.digitize(height, bins) - 1
        if binIndex < len(femaleHist) and femaleHist[binIndex] > maleHist[binIndex]:
            fn += 1
        else:
            tp += 1

    for height in femaleSamples:
        binIndex = np.digitize(height, bins) - 1
        if binIndex < len(maleHist) and maleHist[binIndex] > femaleHist[binIndex]:
            fp += 1
        else:
            tn += 1

    errorRate = ((fn + fp) * 100) / (2 * sample_count)
    confusionMatrix = [[tp, fn], [fp, tn]]
    return errorRate, confusionMatrix