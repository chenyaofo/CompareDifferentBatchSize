import os
import multiprocessing

def init(env):
    os.environ = env

def myfunc():
    print(os.environ['FOO'])


if __name__ == "__main__":
    child_env = os.environ.copy()
    child_env['FOO'] = "foo"
    pool = multiprocessing.Pool(processes=8,initializer=init, initargs=(child_env,))
    pool.apply(myfunc,())