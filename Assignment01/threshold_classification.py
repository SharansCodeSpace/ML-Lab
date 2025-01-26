def findOptimalThreshold(maleSamples, femaleSamples, samplesCount):
    minError = 100
    bestThreshold = 0
    bestConfusionMatrix = None

    for threshold in range(135, 180):
        tp = fn = fp = tn = 0
        for height in maleSamples:
            if height >= threshold:
                tp += 1
            else:
                fn += 1

        for height in femaleSamples:
            if height >= threshold:
                fp += 1
            else:
                tn += 1

        mismatchCount = fn + fp
        errorPer = (mismatchCount * 100) / (samplesCount * 2)

        if errorPer < minError:
            minError = errorPer
            bestThreshold = threshold
            bestConfusionMatrix = [[tp, fn], [fp, tn]]

    return bestThreshold, minError, bestConfusionMatrix