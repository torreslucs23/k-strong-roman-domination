from time import time

def simple_decorator(func):
    def wrapper(*args, **kwargs):
        print("started clock")
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Function took {(end_time - start_time):.4f} seconds to complete.")
        return result
    return wrapper

