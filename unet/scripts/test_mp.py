from multiprocessing import set_start_method, Pool

print("before function definition")

def debug_task(args):
    print(f"Running with args: {args}")
    return args

print("after function definition")

if __name__ == "__main__":
    print("Before set_start_method")
    set_start_method('fork', force=True)
    print("After set_start_method, before with")
    with Pool(4) as pool:
        print("Inside with, before map")
        pool.map(debug_task, range(4))
        print("Inside with, After map")
    print("After with")
