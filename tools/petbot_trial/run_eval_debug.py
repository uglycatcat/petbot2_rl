"""在容器内运行某个 run 副本的 petbot_eval.py，但把静默退出变成可见的 traceback。

petbot_eval.py 的入口块是 `try: main() finally: app.close()`，app.close() 会直接结束进程，
异常 traceback 与控制台输出都会被吞掉。这里去掉入口块、自己调用 main()，异常先打印再退出。
用法（容器内，工作目录=该副本）：
    /isaac-sim/python.sh /workspace/demo_env/run_eval_debug.py <副本内 eval 脚本路径> --headless --num_envs 512 ...
"""

import os
import sys
import traceback

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
source = open(script, encoding="utf-8").read()
source = source.replace('if __name__ == "__main__":', "if False:  # 入口块被包装脚本接管")
module_globals = {"__name__": "__main__", "__file__": script}
try:
    exec(compile(source, script, "exec"), module_globals)
    module_globals["main"]()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
