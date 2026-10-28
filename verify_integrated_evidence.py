"""Verify paper-style natural evidence and separate statistical research evidence."""
import argparse,json
from pathlib import Path
from verify_evidence import verify
from tvmc.statistical_judge import verify_statistical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    def read(name):return json.loads((a.run_dir/name).read_text())
    context=read('public_context.json');natural=read('public_evidence.json')
    result={'natural':verify(context,natural),
            'statistical_research':verify_statistical(context,natural,read('statistical_evidence.json'))}
    print(json.dumps(result,indent=2))
    # Distinguish reading evidence from acceptance of the original paper-style chain.
    return 0 if all(result['natural'].values()) else 2

if __name__=='__main__':raise SystemExit(main())
