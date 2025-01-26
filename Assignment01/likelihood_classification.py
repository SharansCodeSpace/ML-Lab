import statistics

def classifyByLikelihood(maleSamples, femaleSamples, samplesCount):
    maleMeanActual = statistics.mean(maleSamples)
    femaleMeanActual = statistics.mean(femaleSamples)
    maleSDActual = statistics.stdev(maleSamples)
    femaleSDActual = statistics.stdev(femaleSamples)

    maleDist = statistics.NormalDist(mu=maleMeanActual, sigma=maleSDActual)
    femaleDist = statistics.NormalDist(mu=femaleMeanActual, sigma=femaleSDActual)

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for height in maleSamples:
        probMale = maleDist.pdf(height)
        probFemale = femaleDist.pdf(height)
        if probFemale > probMale:
            fn += 1
        else:
            tp += 1

    for height in femaleSamples:
        probMale = maleDist.pdf(height)
        probFemale = femaleDist.pdf(height)
        if probMale > probFemale:
            fp += 1
        else:
            tn += 1

    error = ((fn + fp) * 100) / (samplesCount * 2)
    confusion_matrix = [[tp, fn], [fp, tn]]
    return error, confusion_matrix