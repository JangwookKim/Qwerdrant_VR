import editdistance


def cer(pred, target):

    errors = 0
    total = 0

    for p,t in zip(pred,target):

        errors += editdistance.eval(p,t)
        total += len(t)

    return errors/total


def wer(pred, target):

    errors = 0
    total = 0

    for p,t in zip(pred,target):

        pw = p.split()
        tw = t.split()

        errors += editdistance.eval(pw,tw)
        total += len(tw)

    return errors/total