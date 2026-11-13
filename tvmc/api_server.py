"""Loopback TLS model endpoint for reproducible API/relay experiments.

One actual model instance serves requests; no codeword table is consulted.
This is a local integration harness, not an externally attested service.
"""
import datetime
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import ssl
import threading
import time
from cryptography import x509
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from .evidence import https_query


class Endpoint:
    def __init__(self,directory,lab=None,served_log=None,forward=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False,mode=0o700)
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'TVMC local experiment')])
        now=datetime.datetime.now(datetime.timezone.utc)
        cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
            .not_valid_after(now+datetime.timedelta(days=1)).sign(key,hashes.SHA256()))
        certpath=directory/'cert.pem';keypath=directory/'key.pem'
        certpath.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        keypath.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()));keypath.chmod(0o600)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=4*1024*1024:raise ValueError('request too large')
                    request=json.loads(self.rfile.read(size));probes=request['probes'];client=request['collector']
                    if forward is not None:scores=https_query(forward,probes,client)
                    else:
                        import torch
                        scores=[]
                        with torch.no_grad():
                            for p in probes:
                                if served_log is not None:served_log.record(p['text'],f'peer={self.client_address[0]}:{self.client_address[1]};claimed_collector={client}')
                                z=lab.last_logits(lab.tokens(p['text']))
                                a,b=p['tokens'];scores.append(float(z[a]-z[b]))
                        if served_log is not None:served_log.anchor()
                    body=json.dumps(dict(scores=scores)).encode();self.send_response(200)
                    self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                except Exception as e:
                    body=json.dumps(dict(error=type(e).__name__)).encode();self.send_response(400);self.end_headers();self.wfile.write(body)
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(certpath,keypath)
        self.server.socket=ctx.wrap_socket(self.server.socket,server_side=True)
        self.target=dict(endpoint=f'https://127.0.0.1:{self.server.server_port}/infer',
            tls_sha256=hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest(),start=time.time()-1,end=time.time()+1800)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
