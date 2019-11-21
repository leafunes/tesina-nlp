from logger import log, debug

import pandas as pd

# --------------- Abrir archivos --------------

def star_to_rate(rate):
    if rate > 3:
        return 'positive'
    else:
        return 'negative'

def read_files(base_path, filenames):
    debug("[Leyendo archivos en panda...]")

    finalData = pd.DataFrame(columns=["content", "rate"])
    for fn in filenames:
        data = pd.read_csv( base_path + fn + '.csv')
        finalData = finalData.append(pd.DataFrame({ 'rate': data["RATE"].map(star_to_rate), 'content': data["TITLE"].map(str) + " " + data["CONTENT"] }), sort=False)
    debug("[Archivos Leidos...]")
    return finalData