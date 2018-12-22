from time import time

last_time = {}
sums = {}

def reset_t(tag):
    last_time[tag] = time()
    
def add_t(tag):
    if tag in sums:
        sums[tag] += time() - last_time[tag]
    else:
        sums[tag] = time() - last_time[tag]
    
def clear_t():
    global sums
    sums = {}
    
def log_by_tag(tgt_file = None):
    if tgt_file is None:
        for t in sums:
            print(f"{t}:{sums[t]}")
    else:
        for t in sums:
            tgt_file.write(f"{t}:{sums[t]}\n")

def get(tag):
    return sums[tag]
