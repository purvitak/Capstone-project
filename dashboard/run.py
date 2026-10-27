"""Start the read-only API and Streamlit UI together; Ctrl+C stops both."""
from pathlib import Path
import argparse
import errno
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port",type=int,default=8000)
    parser.add_argument("--ui-port",type=int,default=8501)
    args=parser.parse_args()
    if args.api_port==args.ui_port or any(not 1<=p<=65535 for p in [args.api_port,args.ui_port]):
        parser.error("Choose different valid API/UI ports.")
    for port in [args.api_port,args.ui_port]:
        with socket.socket() as sock:
            try:sock.bind(("127.0.0.1",port))
            except OSError as exc:
                if exc.errno==errno.EADDRINUSE:
                    parser.error(f"Port {port} is busy. Choose another port or stop the existing service.")
                parser.error(f"Cannot listen on port {port}: {exc.strerror}.")
    env=os.environ.copy()
    env["SOLAR_API_URL"]=f"http://127.0.0.1:{args.api_port}"
    env["SOLAR_DATA_MODE"]="api"
    children=[]
    try:
        api=subprocess.Popen([sys.executable,"-m","uvicorn","dashboard.backend.api:app","--host","127.0.0.1","--port",str(args.api_port)],cwd=ROOT,env=env)
        children.append(api)
        for _ in range(200):
            if api.poll() is not None:raise RuntimeError("The forecast API could not start.")
            try:
                with urllib.request.urlopen(env["SOLAR_API_URL"]+"/api/health",timeout=1):break
            except urllib.error.HTTPError as exc:
                if exc.code==503:
                    print("API ready; the archive is unavailable. The dashboard will show a recoverable data message.",flush=True)
                    break
            except (OSError,urllib.error.URLError):pass
            time.sleep(.1)
        else:raise RuntimeError("Timed out waiting for the forecast API.")
        ui=subprocess.Popen([sys.executable,"-m","streamlit","run",str(ROOT/"dashboard/app.py"),"--server.address","127.0.0.1","--server.port",str(args.ui_port),"--server.headless","true","--browser.gatherUsageStats","false"],cwd=ROOT,env=env)
        children.append(ui)
        print(f"\nDashboard: http://127.0.0.1:{args.ui_port}\nAPI docs:  {env['SOLAR_API_URL']}/docs\nCtrl+C stops both services.\n",flush=True)
        while all(child.poll() is None for child in children):time.sleep(.3)
    except KeyboardInterrupt:pass
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
