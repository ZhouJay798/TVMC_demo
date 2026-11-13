"""Download the pinned public corpus without executing downloaded code."""
import argparse,hashlib,urllib.request
from pathlib import Path
URL='https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('.cache/public_data/tinyshakespeare-6f9487a.txt'));args=parser.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest()!='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed':raise ValueError('Existing corpus checksum mismatch')
        print('Verified existing corpus:',args.output);return
    with urllib.request.urlopen(URL,timeout=60) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed':raise ValueError('corpus checksum mismatch')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(data)
    print('Saved',args.output,'sha256',hashlib.sha256(data).hexdigest())
if __name__=='__main__':main()
