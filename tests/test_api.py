import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pytest
from serve import make_server


def test_local_api_budget_and_validation():
    server=make_server(lambda prompts:np.array([len(p)%2 for p in prompts]),port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urlopen(base+'/health'))['one_shot_used'] is False
        def post(data):
            return urlopen(Request(base+'/query',data=json.dumps(data).encode(),
                           headers={'Content-Type':'application/json'}))
        with pytest.raises(HTTPError) as exc: post({'prompts':[]})
        assert exc.value.code==400
        assert json.load(post({'prompts':['a','ab'],'protocol':'batch'}))['bits']==[1,0]
        with pytest.raises(HTTPError) as exc: post({'prompts':['a']})
        assert exc.value.code==409
        assert json.load(urlopen(base+'/health'))['one_shot_used'] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
