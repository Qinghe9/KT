import pandas as pd
from .utils import sta_infos, replace_text

def load_q2c(qname):
    df = pd.read_csv(qname, encoding = "utf-8",low_memory=False).dropna(subset=["name", "topic"])
    dq2c = dict()
    for name, topic in zip(df["name"], df["topic"]):
        if name not in dq2c:
            dq2c[name] = topic
        else:
            print(f"already has topic in dict: {name}: {topic}, {dq2c[name]}")
    print(f"dq2c: {len(dq2c)}")
    return dq2c

KEYS = ["user_id", "topic", "exercise"]
def read_data_from_csv(read_file, write_file, dq2c):
    stares = []

    # The full Junyi log has about 26 million rows.  Reading all 17 source
    # columns creates many gigabytes of unnecessary Python objects, although
    # KT preprocessing only uses the five columns below.  Restricting the
    # input here preserves every interaction while keeping peak RAM practical.
    usecols = ["user_id", "exercise", "time_done",
               "time_taken_attempts", "correct"]
    df = pd.read_csv(
        read_file, usecols=usecols,
        dtype={"user_id": "Int64", "exercise": "string",
               "time_done": "Int64", "time_taken_attempts": "string",
               "correct": "string"},
        low_memory=False)
    df["topic"] = df["exercise"].map(dq2c).fillna("NANA")
    df = df[df["topic"] != "NANA"].dropna(subset=["exercise"])
    df["exercise"] = df["exercise"].apply(replace_text)
    df["topic"] = df["topic"].apply(replace_text)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])
    print(f"original recores shape: {df.shape}")

    usedf = df[["index", "user_id", "exercise", "time_done",
                "time_taken_attempts", "correct", "topic"]].copy()
    usedf = usedf.dropna(subset=["user_id", "exercise", "time_done", "correct"])
    usedf["correct"] = usedf["correct"].str.lower().map(
        {"false": 0, "true": 1})
    usedf = usedf.dropna(subset=["correct"])
    usedf["time_taken_attempts"] = (usedf["time_taken_attempts"].fillna(-100)).astype(str) # only hint! False correct
    usedf.loc[:, "time_taken_attempts"] = usedf["time_taken_attempts"].astype(str).apply(lambda x: int(x.split("&")[0])*1000).astype(str)

    usedf.loc[:, "time_done"] = usedf["time_done"].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(usedf, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    uids = usedf.user_id.unique()
    problems = usedf.exercise.unique()
    print(f"usedf: {usedf.shape}, uids: {len(uids)}, problems: {len(problems)}")

    ui_df = usedf.groupby('user_id', sort=False)

    user_count = 0
    with open(write_file, "w") as fout:
        for uid, curdf in ui_df:
            curdf = curdf.sort_values(by=["time_done", "index"])

            curdf["time_done"] = curdf["time_done"].apply(
                lambda x: round(int(x) / 1000)).astype(str)  # microseconds to milliseconds
            questions = curdf["exercise"].tolist()
            concepts = curdf["topic"].tolist()
            rs = curdf["correct"].astype(int).astype(str).tolist()
            ts = curdf["time_done"].tolist()
            uts = curdf["time_taken_attempts"].tolist()
            rows = [[str(uid), str(len(rs))], questions, concepts, rs, ts, uts]
            for row in rows:
                fout.write(",".join(row) + "\n")
            user_count += 1
            if user_count % 1000 == 0:
                print(user_count)

    print("\n".join(stares))

    return
