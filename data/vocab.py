PAD = "<pad>"
UNK = "<unk>"

def build_char_vocab():

    chars = list("abcdefghijklmnopqrstuvwxyz")
    chars += [" ", "'", ",", ".", ";", "?"]
    chars += list("0123456789")

    vocab = [PAD, UNK] + chars

    char2idx = {c:i for i,c in enumerate(vocab)}
    idx2char = {i:c for c,i in char2idx.items()}

    return char2idx, idx2char


def build_gesture_vocab():

    gestures = ["Q","A","Z","W","X","E","D","C"]
    gestures += [" ", "'", ",", ".", ";", "?"]
    gestures += list("0123456789")

    gestures.append(PAD)

    gesture2idx = {g:i for i,g in enumerate(gestures)}
    idx2gesture = {i:g for g,i in gesture2idx.items()}

    return gesture2idx, idx2gesture