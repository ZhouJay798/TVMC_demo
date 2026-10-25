"""Local one-shot inference API for a trained adapter; no public hosting."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

from tvmc.bridge import OneShotBox


def make_server(predictor, host='127.0.0.1', port=8080):
    box = OneShotBox(predictor)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'ready':True,'one_shot_used':box.used})
            else:
                self.reply(404, {'error':'not found'})

        def do_POST(self):
            if self.path != '/query':
                self.reply(404, {'error':'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 131072:
                    raise ValueError('request must contain 1..131072 bytes')
                data = json.loads(self.rfile.read(length))
                prompts, protocol = data['prompts'], data.get('protocol','batch')
                if (not isinstance(prompts,list) or not 1 <= len(prompts) <= 256 or
                    any(not isinstance(x,str) or len(x)>2048 for x in prompts)):
                    raise ValueError('need 1..256 strings, each at most 2048 characters')
                if protocol not in ('batch','stream','single'):
                    raise ValueError('protocol must be batch, stream, or single')
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error':str(exc)})
                return
            with lock:
                if box.used:
                    self.reply(409, {'error':'one-shot box already consumed; restart server for a new box'})
                    return
                try:
                    bits = box.query(prompts,protocol)
                    self.reply(200, {'bits':np.asarray(bits).tolist(),'protocol':protocol,'requests_used':1})
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.reply(500, {'error':str(exc)})
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--user',type=int,choices=range(1,6),default=1)
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    import os
    os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'.cache'/'huggingface'))
    import torch
    from tvmc.model import ModelLab
    lab=ModelLab(str(args.run_dir/'decrypted_model'),args.run_dir/'cache',args.device,4,True)
    lab.restore(torch.load(args.run_dir/f'user_{args.user}_lora.pt',map_location='cpu',weights_only=True))
    server=make_server(lab.predict,port=args.port)
    print(f'Local API ready: host=127.0.0.1 port={args.port}; GET /health; POST /query; one request per process',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
