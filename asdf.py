import concurrent.futures
import time

def foo(bar):
    return f'foo: {bar}'

def foobar(foo, bar):
    return f'foobar: {foo}, {bar}'

def hello_world():
    time.sleep(5)
    return f'hello world'

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(foo, 'hi')
    future_v2 = executor.submit(foobar, 'hi', 'jon')
    future_v3 = executor.submit(hello_world)
    print(future.result())
    print(future_v2.result())
    time.sleep(3)
    print(future_v3.result())
    print("oh no")
print("oh yes")